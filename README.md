# Credit_Card_Analysis

library(kernlab)
library(kknn)
library(caret)
data<-credit_card_data
summary(data)


### Question 2.2.1

### C=10

set.seed(123)
model <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type="C-svc",kernel="vanilladot",C=10,scaled=TRUE)

a <- colSums(model@xmatrix[[1]] * model@coef[[1]] )

a0 <- -model@b

pred <- predict(model, data[,1:10])

sum(pred == data[,11]) / nrow(data)


### C=20
model2 <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type="C-svc",kernel="vanilladot",C=20,scaled=TRUE)
model2

a <- colSums(model2@xmatrix[[1]] * model2@coef[[1]] )
a

a0 <- -model2@b
a0

pred <- predict(model2, data[,1:10])
pred

sum(pred == data[,11]) / nrow(data)

### C=30

model3 <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type="C-svc",kernel="vanilladot",C=30,scaled=TRUE)
model3

a <- colSums(model3@xmatrix[[1]] * model3@coef[[1]] )
a

a0 <- -model3@b
a0

pred <- predict(model3, data[,1:10])
pred

sum(pred == data[,11]) / nrow(data)

### C=50

model4 <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type="C-svc",kernel="vanilladot",C=50,scaled=TRUE)
model4

a <- colSums(model4@xmatrix[[1]] * model4@coef[[1]] )
a

a0 <- -model4@b
a0

pred <- predict(model4, data[,1:10])
pred

sum(pred == data[,11]) / nrow(data)

### C=100

model7 <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type="C-svc",kernel="vanilladot",C=100,scaled=TRUE)
model7

a <- colSums(model7@xmatrix[[1]] * model7@coef[[1]] )
a

a0 <- -model7@b
a0

pred <- predict(model7, data[,1:10])
pred

sum(pred == data[,11]) / nrow(data)

### C=250

model8 <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type="C-svc",kernel="vanilladot",C=250,scaled=TRUE)
model8

a <- colSums(model8@xmatrix[[1]] * model8@coef[[1]] )
a

a0 <- -model8@b
a0

pred <- predict(model8, data[,1:10])
pred

sum(pred == data[,11]) / nrow(data)

### C=1,000

model10 <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type="C-svc",kernel="vanilladot",C=1000,scaled=TRUE)
model10

a <- colSums(model10@xmatrix[[1]] * model10@coef[[1]] )
a

a0 <- -model10@b
a0

pred <- predict(model10, data[,1:10])
pred

sum(pred == data[,11]) / nrow(data)

### Question 2.2.3

# C=100 for RBF kernel
model11 <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type="C-svc",kernel="rbfdot",C=100,scaled=TRUE)
model11

a <- colSums(model11@xmatrix[[1]] * model11@coef[[1]] )
a

a0 <- -model11@b
a0

pred <- predict(model11, data[,1:10])
pred

sum(pred == data[,11]) / nrow(data)

# Model with C=250 for RBF kernel

model12 <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type="C-svc",kernel="rbfdot",C=250,scaled=TRUE)
model12

a <- colSums(model12@xmatrix[[1]] * model12@coef[[1]] )
a

a0 <- -model12@b
a0

pred <- predict(model12, data[,1:10])
pred

sum(pred == data[,11]) / nrow(data)

### Question 2.2.3

set.seed(123)
chk_acc = function(Z){
  pred<- rep(0,(nrow(data))) 
  
  for (i in 1:nrow(data)){
    knn_model=kknn(V11~.,data[-i,],data[i,],k=Z, scale = TRUE) 
    pred[i] <- as.integer(fitted(knn_model)+0.5)
  }

  acc = sum(pred == data[,11]) / nrow(data)
  return(acc)
}

test_vec <- rep(0,20)
for (Z in 1:20){
  test_vec[Z] = chk_acc(Z) 
}

knn_accuracy <- as.matrix(test_vec * 100) 
knn_accuracy
plot(knn_accuracy, main="K-Nearest Neighbor",xlab="K", ylab="KNN Accuracy")
knn_value <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)

