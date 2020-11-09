#import libraries and dataset
library(tidyverse)
library(skimr)
library(inspectdf)
library(scorecard)
library(caret)
library(h2o)
library(glue)
library(highcharter)
library(data.table)


setwd("C:/Users/Lenovo/Downloads")
raw <- fread("Churn_Modelling.csv")
raw %>% view()
raw %>% glimpse()

#remove unneeded columns
raw <- raw[,4:14]

raw$Exited %>%table() %>% prop.table()



#-------------------DATA PROCESSING------------------------------
raw %>% inspect_na()
df.num <- raw %>%select_if(is.numeric)
df.num <- df.num[,1:8]
df.chr <- raw %>% mutate_if(is.character,as.factor) %>% 
  select_if(is.factor) 




#Outliers
num_vars <- df.num %>% names()
for_vars <- c()

for(b in 1:length(num_vars)){
  outvals <- boxplot(df.num[[num_vars[b]]],plot = F)$out
  if(length(outvals)>0){
    for_vars[b] <- num_vars[b]
  }
}

for_vars <- for_vars %>% as.data.frame() %>% drop_na() %>% pull(.) %>% as.character()
for_vars %>% length()



for (o in for_vars) {
  OutVals <- boxplot(df.num[[o]], plot=F)$out
  mean <- mean(df.num[[o]],na.rm=T)
  
  o3 <- ifelse(OutVals>mean,OutVals,NA) %>% na.omit() %>% as.matrix() %>% t() %>% .[1,]
  o1 <- ifelse(OutVals<mean,OutVals,NA) %>% na.omit() %>% as.matrix() %>% t() %>% .[1,]
  
  val3 <- quantile(df.num[[o]],0.75,na.rm = T) + 1.5*IQR(df.num[[o]],na.rm = T)
  df.num[which(df.num[[o]] %in% o3),o] <- val3
  
  val1 <- quantile(df.num[[o]],0.25,na.rm = T) - 1.5*IQR(df.num[[o]],na.rm = T)
  df.num[which(df.num[[o]] %in% o1),o] <- val1
}


#One Hote encoding
ohe <- dummyVars("~.", data = df.chr) %>% 
  predict(newdata = df.chr) %>% as.data.frame()

df <- cbind(df.num,ohe,raw$Exited) 
names(df) <- names(df) %>% str_replace_all("\\.","_")


#-------------------------------Modelling-------------------------------------

#Weight of evidence

#IV
iv <- df %>% iv(y="V3") %>% as_tibble() %>% 
  mutate(info_value = round(info_value,3)) %>% 
  arrange(desc(info_value))

#Exclude not important variables
ivars <- iv %>% filter(info_value>0.02) %>% select(variable) %>% .[[1]]
df.iv <- df %>% select(V3, ivars)
df.iv %>% dim()


# Exclude not important variables 
ivars <- iv %>% 
  filter(info_value>0.02) %>% 
  select(variable) %>% .[[1]] 

df.iv <- df %>% select(V3,ivars)

df.iv %>% dim()

bins <- df.iv %>% woebin("V3")

dt_list <- df.iv %>% 
  split_df("V3", ratio = 0.8, seed = 123)

train_woe <- dt_list$train %>% woebin_ply(bins) 
test_woe <- dt_list$test %>% woebin_ply(bins)

names <- train_woe %>% names() %>% gsub("_woe","",.)                   
names(train_woe) <- names              ; names(test_woe) <- names
train_woe %>% inspect_na() %>% tail(2) ; test_woe %>% inspect_na() %>% tail(2)



#Multicollinearity-------------------

#coefna
target <- "V3"
features <- train_woe %>% select(-V3) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")
glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

#vif
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")
while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,"variable"]
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = train_woe, family = "binomial")
}

glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features 




#Modelling with GLM
h2o.init()
train_h2o <- train_woe %>% select(target, features) %>% as.h2o()
test_h2o <- test_woe %>% select(target, features) %>% as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial",
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T
)


while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
  }

model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3))

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)

h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)

# ---------------------------- Evaluation Metrices ----------------------------

# Prediction & Confision Matrice
pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)

model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')

eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = dt_list$test$V3 %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")

eva$confusion_matrix$dat

# Check overfitting ----
model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)
