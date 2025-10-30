N = int(input())
scores = list(map(int, input().split()))
sum_of_scores = 0
for score in scores:
    sum_of_scores += score
final_score_n = -sum_of_scores
print(final_score_n)