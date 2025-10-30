# YOUR CODE HERE
A = list(map(int, input().split()))
B = list(map(int, input().split()))

takahashi_score = sum(A)
aoki_score = sum(B)

# To win, Aoki needs to score more than Takahashi's score
# Therefore, the minimum runs Aoki needs to score is Takahashi's score + 1
min_runs_needed = takahashi_score - aoki_score + 1

print(min_runs_needed)