# YOUR CODE HERE
N, X = map(int, input().split())
scores = list(map(int, input().split()))

def calculate_final_grade(all_scores):
    return sum(sorted(all_scores)[1:-1])

min_score = 0
max_score = 100

while min_score <= max_score:
    mid_score = (min_score + max_score) // 2
    all_scores = scores + [mid_score]
    
    if calculate_final_grade(all_scores) >= X:
        max_score = mid_score - 1
    else:
        min_score = mid_score + 1

if min_score > 100:
    print(-1)
else:
    print(min_score)