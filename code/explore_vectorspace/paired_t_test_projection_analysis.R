# https://www.datanovia.com/en/lessons/t-test-assumptions/paired-t-test-assumptions/#:~:text=If%20the%20data%20is%20normally,should%20be%20greater%20than%200.05.&text=From%20the%20output%2C%20the%20two,we%20can%20assume%20the%20normality.
library(tidyverse)
library(ggpubr)
library(rstatix)
library(effsize)
library(dplyr)


# additive CORRECTED VERBS
plain_verb_projections_path <- "PATH-TO/LVC-event-duration/code/io/projection_plain_verbs/semantic_projections.txt"



df <- read.csv(plain_verb_projections_path)
head(df, 3)
df <- df[c("FVC_projection","LVC_projection")]
head(df, 3)

# ---- adding ID to rows: https://www.geeksforgeeks.org/add-index-id-to-dataframe-in-r/
# number of rows in data frame
num_rows = nrow(df)

# creating ID column vector 
ID <- c(1:num_rows)

# binding id column to the data frame
df <- cbind(ID , df)
head(df, 3)

# Transform into long data: 
# gather the before and after values in the same column
df.long <- df %>%
  gather(key = "construction", value = "projection", FVC_projection, LVC_projection)
# adding ID to rows: https://www.geeksforgeeks.org/add-index-id-to-dataframe-in-r/ ----

df <- df  %>% mutate(differences = FVC_projection - LVC_projection)
head(df, 3)

df %>% identify_outliers(differences)

# if n>50, QQplot is preferred over Shapiro test (https://www.datanovia.com/en/lessons/t-test-assumptions/paired-t-test-assumptions/#identify-outliers)
df %>% shapiro_test(differences) 

ggqqplot(df, "differences")

summary(df)

# https://www.sthda.com/english/wiki/paired-samples-t-test-in-r

t.test(df$LVC_projection,df$FVC_projection,paired=TRUE,alternative="two.sided")
t.test(df$LVC_projection,df$FVC_projection,paired=TRUE,alternative="greater")

cohen.d(df$LVC_projection,df$FVC_projection, paired='TRUE')


# https://rpkgs.datanovia.com/ggpubr/reference/ggboxplot.html
ggboxplot(df.long, x="construction", y="projection", add="jitter",title="Semantic projection onto the duration scale")

ggpaired(df.long,  x="construction", y="projection", color="construction", palette="jco",title="Semantic projection onto the duration scale")


