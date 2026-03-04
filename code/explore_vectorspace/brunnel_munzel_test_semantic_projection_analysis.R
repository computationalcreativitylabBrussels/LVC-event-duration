
# https://www.datanovia.com/en/lessons/t-test-assumptions/paired-t-test-assumptions/#:~:text=If%20the%20data%20is%20normally,should%20be%20greater%20than%200.05.&text=From%20the%20output%2C%20the%20two,we%20can%20assume%20the%20normality.
library(tidyverse)
library(ggpubr)
library(rstatix)
library(datarium)
library(dplyr)
library(purrr)
require(stats)
library(stringr)
library(bmtest)
library(utils)


filepaths_df <- read.csv("PATH-TO/LVC-event-duration/code/io/projections_corresponding_filepaths.txt")
base_output_path <- "PATH-TO/io/projection_analysis"

remove_extreme_outliers <- FALSE
remove_outliers <- FALSE
  
test_scores <- list()

csv_test_scores_colnames <- c("LV","nominal","FV","note","test_statistic","p-value","CI-lower","CI-upper","relative_effect","FVC_n","LVC_n")
csv_test_scores <- data.frame(matrix(nrow=0,ncol=length(csv_test_scores_colnames)), stringsAsFactors = FALSE)
colnames(csv_test_scores) <- csv_test_scores_colnames



for (i in 1:nrow(filepaths_df)){
#for (i in 2:3){
  lvp_df <- read.csv(filepaths_df[i, "LVC"], header = FALSE)
  lvp_df['category'] = rep("LVC",nrow(lvp_df))
  fvp_df <- read.csv(filepaths_df[i, "FVC"], header = FALSE)
  fvp_df['category'] = rep("FVC",nrow(fvp_df))
  
  full_verb <- filepaths_df[i,'FV']
  light_verb <- filepaths_df[i,'LV']
  nominal <- filepaths_df[i,'nominal']
  
  # check for outliers
  lvp_outliers <- lvp_df %>% identify_outliers(V1)
  fvp_outliers <- fvp_df %>% identify_outliers(V1)
  
  verb_label <- paste(light_verb,nominal,full_verb, sep="-")
  df_col <- c(light_verb,nominal,full_verb)
  

  
  note <- ""
  
  if (remove_extreme_outliers == TRUE || remove_outliers == TRUE){
    
    if (remove_extreme_outliers == TRUE){
      if (any(lvp_outliers["is.extreme"]) || any(fvp_outliers["is.extreme"])){
        fvp_extremes <- fvp_outliers[fvp_outliers$is.extreme == TRUE,]['V1']
        fvp_df <- fvp_df[(!(fvp_df$V1 %in% fvp_extremes$V1)),]
        
        lvp_extremes <- lvp_outliers[lvp_outliers$is.extreme == TRUE,]['V1']
        lvp_df <- lvp_df[(!(lvp_df$V1 %in% lvp_extremes$V1)),]
        
        note <- "EXTREME OUTLIERS REMOVED"
      } 
    } else if (remove_outliers == TRUE){
      fvp_extremes <- fvp_outliers['V1']
      fvp_df <- fvp_df[(!(fvp_df$V1 %in% fvp_extremes$V1)),]
      
      lvp_extremes <- lvp_outliers['V1']
      lvp_df <- lvp_df[(!(lvp_df$V1 %in% lvp_extremes$V1)),]
      
      note <- "ALL OUTLIERS REMOVED"
  }}
  

  
  if (!remove_extreme_outliers && (any(lvp_outliers["is.extreme"]) || any(fvp_outliers["is.extreme"]))){
    note <- "EXTREME OUTLIERS PRESENT"
  }
  
  combined <- rbind(lvp_df,fvp_df)
  combined$category <- as.factor(combined$category)
  
  if (nrow(fvp_df) < 2 || nrow(lvp_df) < 2){
    df_col <- append(df_col,c('insufficient observations','test_result$statistic','','','','',nrow(fvp_df) ,nrow(lvp_df)))
  } else{
    test_result <- bmtest(formula=V1~category, data = combined, group="category", ci=TRUE, relEff=TRUE, asym = FALSE, randomPerm = TRUE, hypothesis='twoGreater', etl=120)[["bmtest"]][["asDF"]]
    df_col <- append(df_col,c(note,test_result['stat[randomPerm]'],test_result['p[randomPerm]'],test_result['cil[randomPerm]'],test_result['ciu[randomPerm]'],test_result['relEff[randomPerm]'],nrow(fvp_df) ,nrow(lvp_df)))
    test_scores[[verb_label]] <- list(note, test_result)
  }

  csv_test_scores[i,] <- df_col
}

# https://stackoverflow.com/questions/30707112/how-to-save-t-test-result-in-r-to-a-txt-file
make_test_output_string <- function(test_output) {
  return(capture.output(print(test_output)))
}

fileConn <- file(str_glue("{base_output_path}/brunnel_munzel_output.csv"))


test_score_outputs <- c()

for (verb in names(test_scores)){
  s <- make_test_output_string(test_scores[[verb]])
  test_score_outputs <- append(test_score_outputs,c(verb,s,'---------'))
}

writeLines(test_score_outputs,fileConn)
close(fileConn)

output_table_file_name <- str_glue("{base_output_path}/brunner_munzel_table.csv")
write.csv(csv_test_scores,file = output_table_file_name, na='',quote=FALSE,row.names = FALSE)




