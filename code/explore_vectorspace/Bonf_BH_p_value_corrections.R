library(DescTools)

path_to_repo <- YOUR-PATH-TO-REPOSITORY

path_to_brunner_munzel_tables <- "PATH-TO/LVC-event-duration/code/io/projection_analysis/"
filename <- "brunner_munzel_table.csv"

for (path_to_file in list("including_all_outliers/","excluding_all_outliers/","excluding_extreme_outliers/")){
  input_path <- paste(path_to_repo, path_to_brunner_munzel_tables, path_to_file, filename, sep="")
  brunner_munzel_table <- read.csv(input_path)
  
  sorted_brunner_munzel_table <- brunner_munzel_table[order(brunner_munzel_table$p.value),]
  
  sorted_brunner_munzel_table["Bonf-adjusted p-values"] <- p.adjust(sorted_brunner_munzel_table$p.value, method = "bonferroni")
  sorted_brunner_munzel_table["BH-adjusted p-values"] <- p.adjust(sorted_brunner_munzel_table$p.value, method = "BH")
  
  split_input_path <- SplitPath(input_path)
  output_filename <- paste(split_input_path$dirname, "Bf_BH_corrections_", split_input_path$fullfilename, sep="")
  
  write.csv(sorted_brunner_munzel_table, file = output_filename, na='',quote=FALSE,row.names = FALSE)
  
}
