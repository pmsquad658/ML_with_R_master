# Classification and Regression Tree (CART) -------------------------------
# Personal Loan Prediction
ploan <- read.csv("Personal Loan.csv")

# 개인 신용대출 데이터  : 향후에 신용대출을 받을지 아닐지 예측 
# For CART
# CART 지원하는 패키지 
install.packages("party")
library(party)

# For AUROC
# ROC 커브를 계산할 수 있는 패키지 
install.packages("ROCR")
library(ROCR)

ploan.x <- ploan[,-c(1,5,10)]
# 먼저 factor 로 바꿔줘야 에러가 나지 않는다. 
ploan.y <- as.data.frame(as.factor(ploan[,10]))

set.seed(12345)
trn_idx <- sample(1:dim(ploan.y)[1], round(0.7*dim(ploan.y)[1]))

ploan.trn <- cbind(ploan.x[trn_idx,], ploanYN = ploan.y[trn_idx,])
ploan.val <- cbind(ploan.x[-trn_idx,], ploanYN = ploan.y[-trn_idx,])
ploan.all <- rbind(ploan.trn, ploan.val)
# 실무적으로는 all set 을 사용함, 특히 시계열 자료의 경우 

# construct single tree and evaluation
# tree parameter settings
# full tree 까지 커지는 것을 제한
min_criterion = c(0.9, 0.95, 0.99)  # 유의성 신뢰도 테스트 (신뢰수준 90% 이상이면 split 허용, 95%, 99%)
min_split = c(10, 30, 50, 100)  # 현재 영역에 최소한 이만큼은 있어야 분기를 시도해 볼 수 있다.
max_depth = c(0, 10, 5) # root 에서 leaf 까지 노드
tree_result = matrix(0,length(min_criterion)*length(min_split)*length(max_depth),9)
# 열이 9개인 matrix, 각 parameter 조합수만큼 형성 

iter_cnt = 1

# Find the best set of parameters 를 위한 min_criterion, min_split, max_depth 각 조합에 대한 
# 생성값을 tree_result matrix 에 저장 

for (i in 1:length(min_criterion))
{
  for ( j in 1:length(min_split))
  {
    for ( k in 1:length(max_depth))
    {
      
      cat("CART Min criterion:", min_criterion[i], ", Min split:", min_split[j], ", Max depth:", max_depth[k], "\n")
      tmp_control = ctree_control(mincriterion = min_criterion[i], minsplit = min_split[j], maxdepth = max_depth[k])
      tmp_tree <- ctree(ploanYN ~ ., data = ploan.trn, controls = tmp_control) # 앞선 줄에서 설정한 옵션을 쓰겠다.
      tmp_tree_val_prediction <- predict(tmp_tree, newdata = ploan.val)
      tmp_tree_val_response <- treeresponse(tmp_tree, newdata = ploan.val) # class label 값을 내어달라
      tmp_tree_val_prob <- 1-unlist(tmp_tree_val_response, use.names=F)[seq(1,nrow(ploan.val)*2,2)] 
      # 각각의 범주에 속할 확률을 내어달라 (확률값)
      tmp_tree_val_rocr <- prediction(tmp_tree_val_prob, ploan.val$ploanYN)  
      
      tmp_tree_val_cm <- table(ploan.val$ploanYN, tmp_tree_val_prediction) # confusion matrix 를 생성
      
      # parameters
      tree_result[iter_cnt,1] = min_criterion[i]
      tree_result[iter_cnt,2] = min_split[j]
      tree_result[iter_cnt,3] = max_depth[k]
      # Recall
      Recall = tmp_tree_val_cm[2,2]/(tmp_tree_val_cm[2,1]+tmp_tree_val_cm[2,2])
      tree_result[iter_cnt,4] = Recall
      # Precision
      Precision <- tmp_tree_val_cm[2,2]/(tmp_tree_val_cm[1,2]+tmp_tree_val_cm[2,2])
      tree_result[iter_cnt,5] = Precision
      # Accuracy
      tree_result[iter_cnt,6] = (tmp_tree_val_cm[1,1]+tmp_tree_val_cm[2,2])/sum(tmp_tree_val_cm)
      # F1 measure
      tree_result[iter_cnt,7] = 2*Recall*Precision/(Recall+Precision)
      # AUROC
      tree_result[iter_cnt,8] = unlist(performance(tmp_tree_val_rocr, "auc")@y.values)
      # Number of leaf nodes
      tree_result[iter_cnt,9] = length(nodes(tmp_tree, unique(where(tmp_tree))))
      iter_cnt = iter_cnt + 1
    }
  }
}

# Find the best set of parameters
tree_result # tree_result matrix 확인 
# Recall, # Precision, # Accuracy, # F1 measure, # AUROC, # Number of leaf nodes 순으로 출력 

tree_result <- tree_result[order(tree_result[,8], decreasing = T),] # AUROC 가 가장 큰 것으로 채택
best_criterion <- tree_result[1,1]
best_split <- tree_result[1,2]
best_depth <- tree_result[1,3]

# Construct the best tree
tree_control = ctree_control(mincriterion = best_criterion, minsplit = best_split, maxdepth = best_depth)
tree <- ctree(ploanYN ~ ., data = ploan.all, controls = tree_control)
  # ploan.all 을 쓴다는 점에 주목. 교과서적이진 않지만 실무적으로 예측이 실제에 최대한 가깝게 하기 위해
  # 이렇게 최종적으로는 all model 을 쓴다. 
  # 여기의 경우 val_set 은 best parameter 를 찾기 위해 쓰이고 실제 분류모델 구축은 all_set 으로 진행
  # 회귀랑은 val의 쓰임이 달랐음 
tree_all_prediction <- predict(tree, newdata = ploan.all)
# [1] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0
tree_all_response <- treeresponse(tree, newdata = ploan.all)
# [[986]]  [[989]]
# [1] 1 0  [1] 0.7758621 0.2241379
# class label 값으로 내어달라 : dummy 코딩 값이 나옴 

tree_all_prob <- 1-unlist(tree_all_response, use.names=F)[seq(1,nrow(ploan.all)*2,2)]
# 위의 값 중 names 빼고 성공의 확률값으로 반환 
tree_all_rocr <- prediction(tree_all_prob, ploan.all$ploanYN)
# 성공의 확률값과, 실제값과 비교 

# Performance of the best tree
# Confusion matrix
tree_all_cm <- table(ploan.all$ploanYN, tree_all_prediction)

# best_result matrix 작성 : 1 by 6
best_result <- matrix(0,1,6) # 빈 공간 만들고 : 0을 1행 6열 

# Recall
Recall = tree_all_cm[2,2]/(tree_all_cm[2,1]+tree_all_cm[2,2])
best_result[1,1] = Recall
# Precision
Precision <- tree_all_cm[2,2]/(tree_all_cm[1,2]+tree_all_cm[2,2])
best_result[1,2] = Precision
# Accuracy
best_result[1,3] = (tree_all_cm[1,1]+tree_all_cm[2,2])/sum(tree_all_cm)
# F1 measure
best_result[1,4] = 2*Recall*Precision/(Recall+Precision)
# AUROC
best_result[1,5] = unlist(performance(tree_all_rocr, "auc")@y.values) # roc 커브의 아래 면적 구하는 식 
# Number of leaf nodes
best_result[1,6] = length(nodes(tree, unique(where(tree))))

# best_result matrix 값 확인 
best_result


# Plot the ROC
tmp <- 1-unlist(tree_all_response, use.names=F)[seq(1,nrow(ploan.all)*2,2)]
tmp.rocr <- prediction(tmp, ploan.all$ploanYN)
tmp.perf <- performance(tmp.rocr, "tpr","fpr")
plot(tmp.perf, col=5, lwd = 3)

# Plot the best tree
plot(tree)
plot(tree, type="simple")  # 단순트리 

# Print rules
print(tree) # tree가 어느 포인트에서 구분되었는지. 

## MEMO ##--------------------------------------------------------------------------------
# best parameter 를 찾는 것은 val set 으로 시행함

