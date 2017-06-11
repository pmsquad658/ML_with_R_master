# Association Rules -------------------------------------------------------
# arules and arulesViz packages install

setwd("C:/R/machine learning with R master/Week03_Association Rule Mining")
install.packages("arules", dependencies = TRUE)      # 연관분석 패키지 
install.packages("arulesViz", dependencies = TRUE)   # 시각화 기능을 추가하는 패키지 
install.packages("wordcloud", dependencies = TRUE)

library(arules)
library(arulesViz)
library(wordcloud)

# Load titanic data set
titanic <- read.delim("titanic.txt", dec=",")   # 특정한 형태의 구분자를 불러들일 때 
                                                # tab 이 default 임 
str(titanic)
head(titanic)

# factor 는 먼저 character 로 변환 후 숫자 변환 해야 한다는 내용 
a <- c(1,3,4,10)
b <- as.factor(a)
b
as.numeric(b)
b
as.character(b)
b
as.numeric(as.character(b))


# Remove "Name" column and group "Age" column
titanic_ar <- titanic[,2:5]                     # tatanic[,-1]
titanic_ar$Age = as.character(titanic_ar$Age)   # factor 를 문자열로 우선 변경해야 함 
c_idx <- which(as.numeric(titanic_ar$Age) < 20) # 인덱스 뽑아내는 함수 
a_idx <- which(as.numeric(titanic_ar$Age) >= 20)
na_idx <- which(is.na(titanic_ar$Age))

titanic_ar$Age[c_idx] <- "Child"
titanic_ar$Age[a_idx] <- "Adult"
titanic_ar$Age[na_idx] <- "Unknown"

# Convert the attribues to factor    # 연관규칙 분석은 모두 요인 형태로 바꾸어야 한다 
titanic_ar$Age <- as.factor(titanic_ar$Age)
titanic_ar$Survived <- as.factor(titanic_ar$Survived)

# Rule generation by Apriori algorithm with default settings
rules <- apriori(titanic_ar)
inspect(rules)
?apriori

# Rule generation by Apriori algorithm with custom settings
rules <- apriori(titanic_ar, parameter = list(minlen = 3, support = 0.1, conf = 0.8),
                 appearance = list(rhs = c("Survived=0", "Survived=1"), default="lhs"))
inspect(rules)

# minlen : 최소 사용 아이템 수 
# appearance : 나오는 것을 control 할 수 있음 

# Plot the rules
plot(rules, method="scatterplot")   # 각 규칙들의 분포 형태를 scatterplot 으로 제고
plot(rules, method="graph", control=list(type = "items", alpha = 1))
# 원의 크기는 지지도, 원의 색상은 향상도 의미 / 신뢰도 정보는 빠져있음 

plot(rules, method="paracoord", control=list(reorder=TRUE))
# Position 이 3,2,1 이면 rhs 이다 
# 여기도 신뢰도는 빠져 있음 

#++++Groceries Data +++++++++++++++++++++++++++++++++++++++++++++++++++++

# Load transaction data "Groceries"
data("Groceries")      # 패키지 내에 데이터가 불러들여짐 
summary(Groceries)
str(Groceries)
inspect(Groceries)

# Item inspection
itemName <- itemLabels(Groceries)
itemCount <- itemFrequency(Groceries)*9835

col <- brewer.pal(8, "Dark2")   # 색상조합 템플릿 
wordcloud(words = itemName, freq = itemCount, min.freq = 1, scale = c(7, 0.2), col = col , random.order = FALSE)
wordcloud(words = itemName, freq = itemCount, min.freq = 3, scale = c(3, 0.2), col = col , random.order = FALSE)
itemFrequencyPlot(Groceries, support = 0.05, cex.names=0.8)

# Rule generation by Apriori
rules <- apriori(Groceries, parameter=list(support=0.001, confidence=0.5))
rules

# List the first three rules with the highest lift values
inspect(head(sort(rules, by="lift"),3))
# lift 큰 순서로 정렬하고 싶을 때 

# Save the rules in a text file
write.csv(as(rules, "data.frame"), "Groceries_rules.csv", row.names = FALSE)

# Plot the rules
plot(rules)
plot(rules, method="grouped") 
# 버블차트형식, 정보가 많이 누락되긴 했음 



################# Memo ##############################
# 기본셋팅 최소지지도는 디폴트 0.1 임
# support(지지도) : frequent 에 대한 확률빈도
# confidence(신뢰도) : A 가 일어날 때 / A,B : 조건부 확률 
# lift : 서로 독립적일 때 대비 일어날 빈도 배수를 보임 : 연관이있어서 발생/우연에 의해 발생 
#        1보다 커야 연관이 있음 (1일때는 독립사건)
# A priori algorithm : 최소지지도 조건을 만족치 않는 superset 은 버림 
# 지지도, 신뢰도, 향상도가 모두 클 경우에만 효과성 척도를 비교 가능 (목적에 따라 parameter 활용 다름) 
