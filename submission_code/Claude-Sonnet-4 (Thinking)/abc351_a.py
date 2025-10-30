# Read Team Takahashi's scores
A = list(map(int, input().split()))

# Read Team Aoki's scores (through 8 innings)
B = list(map(int, input().split()))

# Calculate total scores
takahashi_total = sum(A)
aoki_total_through_8 = sum(B)

# Calculate minimum runs needed to win
min_runs_needed = takahashi_total - aoki_total_through_8 + 1

print(min_runs_needed)