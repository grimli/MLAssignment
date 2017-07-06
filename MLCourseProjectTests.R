setwd("/home/bompiani/Coursera/courses/08_PracticalMachineLearning/MLAssignment/")
library(caret)

dfTrain <- read.csv("pml-training.csv")
dfTest <- read.csv("pml-testing.csv")

# elimino ariabili a varianza zero
dfTrain <- dfTrain <- dfTrain[,-nearZeroVar(dfTrain)]

# elimino misurazioni incomplete
dfTrain <- dfTrain[, complete.cases(t(dfTrain))]

# elimino le variabili che sono etichette 
dfTrain <- dfTrain[, -grep("X", names(dfTrain))]
dfTrain <- dfTrain[, -grep("user_name", names(dfTrain))]

# elimino la sequenza temporale
# gli esercizi potrebbero essere stati esguiti in un dato ordine
# non voglio che la stima sia influenzata dai tempi di esecuzione ma solo dai tipi di movimento
dfTrain <- dfTrain[, -grep("timestamp", names(dfTrain))]

# suddivido il campione in due gruppi: 70% train 30% confronto modelli 
set.seed(666)
inTrain <- createDataPartition(y=dfTrain$classe, p=.7, list=FALSE)
dfTrainT <- dfTrain[inTrain, ]
dfTrainV <- dfTrain[-inTrain, ]

# pulisco anche i dati di Test
variables <- names(dfTrain)
dfTest <- dfTest[, variables ]

# parto con le analisi
## Random Tree
fitRT <- train( classe ~ ., data = dfTrainT, method = "rpart")
predRT <- predict( fitRT, newdata = dfTrainV )
cmRT <- confusionMatrix(predRT, dfTrainV$classe)
cmRT

## Generalized partial least square
fitLM <- train( classe ~ ., data = dfTrainT, method = "gpls")
predLM <- predict( fitLM, newdata = dfTrainV )
cmLM <- confusionMatrix(predLM, dfTrainV$classe)
cmLM

## Random Forest
fitRF <- train( classe ~ ., data = dfTrainT, method = "rf")
predRF <- predict( fitRF, newdata = dfTrainV )
cmRF <- confusionMatrix(predRF, dfTrainV$classe)
cmRF

## Boosting
fitBT <- train( classe ~ ., data = dfTrainT, method = "gbm", verbose = FALSE)
predBT <- predict( fitBT, newdata = dfTrainV )
cmBT <- confusionMatrix(predBT, dfTrainV$classe)
cmBT

## Linear Discriminant analysis
fitLDA <- train( classe ~ ., data = dfTrainT, method = "lda")
predLDA <- predict( fitLDA, newdata = dfTrainV )
cmLDA <- confusionMatrix(predLDA, dfTrainV$classe)
cmLDA

## combinate
predDF <- data.frame(predRF,predBT, predLDA, classe=dfTrainV$classe)
fitComb <- train(classe ~ . , data = predDF, method="rf")
predComb <- predict(fitComb, dfTrainV)
cmComb <- confusionMatrix(predComb, dfTrainV$classe)
cmComb

cmRT$overall
cmLM$overall
cmRF$overall
cmBT$overall
cmLDA$overall
cmComb$overall


# approccio dettagliato non necessario
# solo le variabili complete

# 1 - molti campi sembrano contenere solo NA. Da verificare ma Eliminabili 
# 2 - timestamps da capire meglio: non voglio che la previsione dipenda dal momento dellésecuzione
# 3 - X cos'è ? Dovrebbe essere semplicemente lÍD della misura -> da non considerare nel modello
# 4 - window come timestamp
# 5 - diverse variabili total. Non le userei insieme alle parziali da cui dovrebbe essere 
#       completamente determinata. Potrei invece confrontare due modelli uno con le total ed uno con le variabili di dettaglio

# 6 - username attenzione da capire se ad ognuno hanno chiesto di fare uno specifico errore
ggplot(data = dfTrain)+geom_point(aes(x=user_name, y=classe, col=user_name))
# questo grafico mostra che ogni persona ha fatto esercizi in tutte le classi


variables <- names(dfTrain)

## Sensori
# arm
# forearm names(dfTrain[, grep("forearm", names(dfTrain))])
# belt
# dumbbell
names_forearm <- names(dfTrain[, grep("forearm", names(dfTrain))])
dfTrain_forearm <- dfTrain[,names_forearm]
dfTrain_tmp <- dfTrain[,  -grep("forearm", names(dfTrain))]

names_dumbbell <- names(dfTrain_tmp[, grep("dumbbell", names(dfTrain_tmp))])
dfTrain_dumbbell <- dfTrain_tmp[,names_dumbbell]
dfTrain_tmp <- dfTrain_tmp[,  -grep("dumbbell", names(dfTrain_tmp))]

names_belt <- names(dfTrain_tmp[, grep("belt", names(dfTrain_tmp))])
dfTrain_belt <- dfTrain_tmp[,names_belt]
dfTrain_tmp <- dfTrain_tmp[,  -grep("belt", names(dfTrain_tmp))]

names_arm <- names(dfTrain_tmp[, grep("arm", names(dfTrain_tmp))])
dfTrain_arm <- dfTrain_tmp[,names_arm]
dfTrain_tmp <- dfTrain_tmp[,  -grep("arm", names(dfTrain_tmp))]

head(dfTrain_tmp)
# in questo modo si vede che ci sono 38 misure per ognuno dei 5 sensori e che le altre non dovrebbero entrare nella stima
# X è un numero progressivo che individua la misura - OK
# user_name - 6 persone per mediare gli effetti. Motiverei comunque dicendo che stiamo cercando un modello di interesse generale 
# time stamp da ignorare perchè il quando sono state fatte le misure non dovrebbe essere influente 
# window ?
# Classe è il valore che voglio stimare


# da motivare ma considererei solo:
# 1 - roll_*, pithc_*, yaw_* : da coapire meglio cosa sono ma sembrano valori misurati
# 2 - gyros_*, accel_*, magnet_* :
# in sostanz quelle che sono misure dirette dei sensori

# non considereri invece
# - total_* funzioni univoche dei valori parziali
# - kurtosis_* da capire meglio
# - skewness_* 
# - max_*
# - min_*
# - amplitude_
# - var 


# Verifico che non ho variabili a varianza nulla
names(dfTrain[,nearZeroVar(dfTrain)])
names_armc <- names(dfTrain_arm[,nearZeroVar(dfTrain_arm)])

dfTrain_armc <- dfTrain_arm[,-nearZeroVar(dfTrain_arm)]
dfTrain_forearmc <- dfTrain_arm[,-nearZeroVar(dfTrain_arm)]
dfTrain_beltc <- dfTrain_arm[,-nearZeroVar(dfTrain_arm)]
dfTrain_dumbbellc <- dfTrain_arm[,-nearZeroVar(dfTrain_arm)]

# non è necessario escludere variabili a zero varianza
nearZeroVar(dfTrain_armc)
nearZeroVar(dfTrain_forearmc)
nearZeroVar(dfTrain_beltc)
nearZeroVar(dfTrain_dumbbellc)


# elimino le misurazioni non complete
dfTrain_armc_complete <- dfTrain_armc[, complete.cases(t(dfTrain_armc))]
dfTrain_forearmc_complete <- dfTrain_forearmc[, complete.cases(t(dfTrain_forearmc))]
dfTrain_beltc_complete <- dfTrain_beltc[, complete.cases(t(dfTrain_beltc))]
dfTrain_dumbbellc_complete <- dfTrain_dumbbellc[, complete.cases(t(dfTrain_dumbbellc))]

names_armc <- names(dfTrain_armc_complete)
names_forearmc <- names(dfTrain_forearmc_complete)
names_beltc <- names(dfTrain_beltc_complete)
names_dumbbellc <- names(dfTrain_dumbbellc_complete)

dfTrainC <- dfTrain[, c("classe", names_beltc,names_armc, names_forearmc, names_dumbbellc )]

