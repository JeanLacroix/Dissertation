options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
} else {
  getwd()
}

data_dir <- file.path(script_dir, "data")
output_dir <- file.path(script_dir, "output")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

sample_path <- file.path(data_dir, "comps_sample.csv")
other_europe_path <- file.path(data_dir, "other_europe_bucket_composition.csv")

if (!file.exists(sample_path)) {
  stop("Missing input file: ", sample_path)
}

if (!file.exists(other_europe_path)) {
  stop("Missing input file: ", other_europe_path)
}

sample <- read.csv(sample_path, check.names = FALSE)
other_europe <- read.csv(other_europe_path, check.names = FALSE)

if (nrow(sample) == 0) {
  stop("The sample CSV is empty.")
}

palette <- list(
  ink = "#1F2933",
  muted = "#52616B",
  grid = "#D8DEE4",
  cool = "#2A6F97",
  cool_light = "#98C1D9",
  warm = "#D1495B",
  gold = "#E9C46A",
  sand = "#F6F1EB",
  slate = "#5E6472"
)

open_png <- function(filename, width = 10, height = 7) {
  png(
    filename = file.path(output_dir, filename),
    width = width,
    height = height,
    units = "in",
    res = 170,
    bg = "white"
  )
}

close_png <- function() {
  invisible(dev.off())
}

add_top_titles <- function(main_text, subtitle_text) {
  mtext(main_text, side = 3, line = 2.4, cex = 1.52, font = 2, col = palette$ink)
  mtext(subtitle_text, side = 3, line = 0.9, cex = 1, col = palette$muted)
}

share_labels <- function(values, total) {
  sprintf("%d (%.1f%%)", as.integer(values), 100 * as.numeric(values) / total)
}

ordered_levels <- function(data, label_col, order_col) {
  lookup <- unique(data[, c(label_col, order_col)])
  lookup <- lookup[order(lookup[[order_col]]), , drop = FALSE]
  lookup[[label_col]]
}

format_cell <- function(value, digits = 0, suffix = "") {
  paste0(formatC(value, format = "f", digits = digits), suffix)
}

pick_fill <- function(value, min_value, max_value, fill_scale) {
  if (max_value == min_value) {
    return(fill_scale[length(fill_scale)])
  }

  scaled <- (value - min_value) / (max_value - min_value)
  index <- floor(scaled * (length(fill_scale) - 1)) + 1
  fill_scale[pmax(1, pmin(length(fill_scale), index))]
}

draw_heatmap <- function(mat, filename, title_text, subtitle_text, footnote_text, digits = 0, suffix = "") {
  nr <- nrow(mat)
  nc <- ncol(mat)
  min_value <- min(mat)
  max_value <- max(mat)
  fill_scale <- grDevices::colorRampPalette(
    c(palette$sand, palette$cool_light, palette$cool, "#153B50")
  )(100)

  open_png(filename, width = 11, height = 8)
  par(
    mar = c(12.5, 14, 7, 2),
    family = "sans",
    col.axis = palette$ink,
    col.lab = palette$ink,
    fg = palette$ink,
    bty = "n"
  )

  plot(
    c(0, nc), c(0, nr),
    type = "n",
    axes = FALSE,
    xlab = "",
    ylab = "",
    xaxs = "i",
    yaxs = "i"
  )

  row_order <- rev(seq_len(nr))

  for (i in seq_len(nr)) {
    row_index <- row_order[i]
    for (j in seq_len(nc)) {
      value <- mat[row_index, j]
      fill <- pick_fill(value, min_value, max_value, fill_scale)
      scaled <- if (max_value == min_value) 1 else (value - min_value) / (max_value - min_value)
      text_col <- if (scaled > 0.58) "white" else palette$ink

      rect(j - 1, i - 1, j, i, col = fill, border = "white", lwd = 2)
      text(
        x = j - 0.5,
        y = i - 0.5,
        labels = format_cell(value, digits = digits, suffix = suffix),
        cex = 0.82,
        col = text_col
      )
    }
  }

  axis(
    side = 1,
    at = seq(0.5, nc - 0.5, by = 1),
    labels = colnames(mat),
    tick = FALSE,
    las = 2,
    cex.axis = 0.9
  )

  axis(
    side = 2,
    at = seq(0.5, nr - 0.5, by = 1),
    labels = rev(rownames(mat)),
    tick = FALSE,
    las = 2,
    cex.axis = 0.9
  )

  box(col = palette$grid)
  add_top_titles(title_text, subtitle_text)
  mtext(footnote_text, side = 1, line = 10.6, adj = 0, cex = 0.84, col = palette$muted)
  close_png()
}

total_rows <- nrow(sample)

country_counts_full <- sort(table(sample$country), decreasing = TRUE)
country_counts_plot <- sort(country_counts_full)
asset_counts_full <- sort(table(sample$asset_type), decreasing = TRUE)
asset_counts_plot <- sort(asset_counts_full)
year_counts <- table(sample$transaction_year)

uk_share <- 100 * as.numeric(country_counts_full["United Kingdom"]) / total_rows
uk_other_share <- 100 * (
  as.numeric(country_counts_full["United Kingdom"]) +
    as.numeric(country_counts_full["Other Europe"])
) / total_rows

top_three_asset_share <- 100 * sum(head(asset_counts_full, 3)) / total_rows
year_2021_2022_share <- 100 * sum(year_counts[names(year_counts) %in% c("2021", "2022")]) / total_rows

size_levels <- ordered_levels(sample, "size_bucket", "size_bucket_order")
price_levels <- ordered_levels(sample, "price_bucket", "price_bucket_order")
price_per_sqm_levels <- ordered_levels(sample, "price_per_sqm_bucket", "price_per_sqm_bucket_order")
asset_levels <- names(asset_counts_full)

size_price_matrix <- table(
  factor(sample$size_bucket, levels = size_levels),
  factor(sample$price_bucket, levels = price_levels)
)

asset_value_density_matrix <- table(
  factor(sample$asset_type, levels = asset_levels),
  factor(sample$price_per_sqm_bucket, levels = price_per_sqm_levels)
)

asset_value_density_pct <- round(prop.table(asset_value_density_matrix, 1) * 100, 0)

mid_size_levels <- c("2,500-5,000 sqm", "5,000-10,000 sqm", "10,000-25,000 sqm")
mid_price_levels <- c("10-25 EUR mn", "25-50 EUR mn", "50-100 EUR mn")
mid_core_share <- 100 * sum(
  size_price_matrix[rownames(size_price_matrix) %in% mid_size_levels,
                    colnames(size_price_matrix) %in% mid_price_levels]
) / total_rows

high_density_levels <- c(
  "5,000-7,500 EUR/sqm",
  "7,500-10,000 EUR/sqm",
  "10,000-15,000 EUR/sqm",
  "15,000-25,000 EUR/sqm",
  "25,000+ EUR/sqm"
)

high_density_share <- tapply(
  sample$price_per_sqm_bucket %in% high_density_levels,
  sample$asset_type,
  mean
) * 100

top5_other_europe_share <- 100 * sum(head(other_europe$transactions, 5)) / sum(other_europe$transactions)

open_png("01_country_concentration.png", width = 10, height = 7)
par(
  mar = c(6.5, 9.5, 7, 2),
  family = "sans",
  col.axis = palette$ink,
  col.lab = palette$ink,
  fg = palette$ink,
  bty = "n"
)
country_cols <- rep(palette$cool_light, length(country_counts_plot))
names(country_cols) <- names(country_counts_plot)
country_cols["Other Europe"] <- palette$gold
country_cols["United Kingdom"] <- palette$warm
country_bar_pos <- barplot(
  country_counts_plot,
  horiz = TRUE,
  las = 1,
  col = country_cols,
  border = NA,
  xlim = c(0, max(country_counts_plot) * 1.35),
  xlab = "Transactions"
)
text(
  x = as.numeric(country_counts_plot) + max(country_counts_plot) * 0.03,
  y = country_bar_pos,
  labels = share_labels(country_counts_plot, total_rows),
  adj = 0,
  cex = 0.88,
  col = palette$ink,
  xpd = TRUE
)
add_top_titles(
  "Country concentration is the main structural risk",
  "The sample is dominated by United Kingdom rows and a pooled Other Europe bucket."
)
mtext(
  sprintf("United Kingdom = %.1f%% of rows. United Kingdom + Other Europe = %.1f%%.", uk_share, uk_other_share),
  side = 1,
  line = 4.2,
  adj = 0,
  cex = 0.86,
  col = palette$muted
)
close_png()

open_png("02_asset_coverage.png", width = 10, height = 7)
par(
  mar = c(6.5, 10, 7, 2),
  family = "sans",
  col.axis = palette$ink,
  col.lab = palette$ink,
  fg = palette$ink,
  bty = "n"
)
asset_cols <- rep(palette$cool, length(asset_counts_plot))
asset_cols[asset_counts_plot < 30] <- palette$warm
asset_cols[asset_counts_plot >= 30 & asset_counts_plot < 100] <- palette$gold
asset_bar_pos <- barplot(
  asset_counts_plot,
  horiz = TRUE,
  las = 1,
  col = asset_cols,
  border = NA,
  xlim = c(0, max(asset_counts_plot) * 1.35),
  xlab = "Transactions"
)
abline(v = 30, col = palette$slate, lty = 2, lwd = 1.2)
text(
  x = as.numeric(asset_counts_plot) + max(asset_counts_plot) * 0.03,
  y = asset_bar_pos,
  labels = share_labels(asset_counts_plot, total_rows),
  adj = 0,
  cex = 0.88,
  col = palette$ink,
  xpd = TRUE
)
add_top_titles(
  "Coverage is deep for only a few asset classes",
  "Thin buckets should be read as directional evidence, not stable segments."
)
mtext(
  sprintf("Office, Industrial and Retail make up %.1f%% of all rows. The dashed line marks a 30-row fragility threshold.", top_three_asset_share),
  side = 1,
  line = 4.2,
  adj = 0,
  cex = 0.86,
  col = palette$muted
)
close_png()

open_png("03_year_coverage.png", width = 10, height = 7)
par(
  mar = c(7, 4.5, 7, 2),
  family = "sans",
  col.axis = palette$ink,
  col.lab = palette$ink,
  fg = palette$ink,
  bty = "n"
)
year_cols <- rep(palette$cool_light, length(year_counts))
names(year_cols) <- names(year_counts)
year_cols[names(year_counts) %in% c("2021", "2022")] <- palette$cool
year_cols[names(year_counts) == "2026"] <- palette$warm
year_bar_pos <- barplot(
  year_counts,
  col = year_cols,
  border = NA,
  ylim = c(0, max(year_counts) * 1.18),
  ylab = "Transactions",
  xlab = "Transaction year"
)
text(
  x = year_bar_pos,
  y = as.numeric(year_counts) + max(year_counts) * 0.03,
  labels = share_labels(year_counts, total_rows),
  cex = 0.88,
  col = palette$ink
)
add_top_titles(
  "The sample is concentrated in 2021-2022, not evenly through time",
  "Recent-year coverage thins out sharply, especially at the 2026 edge."
)
mtext(
  sprintf("Rows from 2021-2022 account for %.1f%% of the sample, while 2026 contains only %d observations.", year_2021_2022_share, as.integer(year_counts["2026"])),
  side = 1,
  line = 5,
  adj = 0,
  cex = 0.86,
  col = palette$muted
)
close_png()

draw_heatmap(
  mat = size_price_matrix,
  filename = "04_size_price_heatmap.png",
  title_text = "Most observations sit in the middle of the size-value spectrum",
  subtitle_text = "The tails exist, but the sample center is clearly mid-size and mid-ticket.",
  footnote_text = sprintf("Rows in the 2,500-25,000 sqm and EUR 10m-100m corridor make up %.1f%% of the sample.", mid_core_share)
)

draw_heatmap(
  mat = asset_value_density_pct,
  filename = "05_value_density_by_asset.png",
  title_text = "Value density differs sharply by asset type",
  subtitle_text = "Office skews into higher EUR/sqm buckets, while Industrial remains concentrated at the low end.",
  footnote_text = sprintf("Share above EUR 5,000/sqm: Office %.1f%% versus Industrial %.1f%%.", high_density_share["Office"], high_density_share["Industrial"]),
  digits = 0,
  suffix = "%"
)

open_png("06_other_europe_bucket.png", width = 10, height = 8)
par(
  mar = c(6.8, 10.5, 7, 2),
  family = "sans",
  col.axis = palette$ink,
  col.lab = palette$ink,
  fg = palette$ink,
  bty = "n"
)
other_europe_plot <- other_europe[order(other_europe$transactions), , drop = FALSE]
bucket_cols <- rep(palette$cool_light, nrow(other_europe_plot))
bucket_cols[seq.int(from = nrow(other_europe_plot) - 4, to = nrow(other_europe_plot))] <- palette$warm
bucket_bar_pos <- barplot(
  other_europe_plot$transactions,
  names.arg = other_europe_plot$country,
  horiz = TRUE,
  las = 1,
  col = bucket_cols,
  border = NA,
  xlim = c(0, max(other_europe_plot$transactions) * 1.28),
  xlab = "Transactions inside the pooled bucket"
)
text(
  x = other_europe_plot$transactions + max(other_europe_plot$transactions) * 0.03,
  y = bucket_bar_pos,
  labels = other_europe_plot$transactions,
  adj = 0,
  cex = 0.86,
  col = palette$ink,
  xpd = TRUE
)
add_top_titles(
  "Other Europe is a pooled region, not a coherent single market",
  "The bucket spans many countries with thin individual coverage."
)
mtext(
  sprintf("This pooled group contains %d countries. Its top five contributors account for %.1f%% of the bucket.", nrow(other_europe), top5_other_europe_share),
  side = 1,
  line = 4.2,
  adj = 0,
  cex = 0.86,
  col = palette$muted
)
close_png()

generated_files <- c(
  "01_country_concentration.png",
  "02_asset_coverage.png",
  "03_year_coverage.png",
  "04_size_price_heatmap.png",
  "05_value_density_by_asset.png",
  "06_other_europe_bucket.png"
)

cat("Generated", length(generated_files), "charts in", normalizePath(output_dir, winslash = "/", mustWork = FALSE), "\n")
cat(paste(generated_files, collapse = "\n"), "\n")
