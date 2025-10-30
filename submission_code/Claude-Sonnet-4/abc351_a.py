# Read input
takahashi_scores = list(map(int, input().split()))
aoki_scores = list(map(int, input().split()))

# Calculate total scores
takahashi_total = sum(takahashi_scores)
aoki_current = sum(aoki_scores)

# Team Aoki needs to score strictly more than Team Takahashi
# aoki_current + runs_needed > takahashi_total
# runs_needed > takahashi_total - aoki_current
runs_needed = takahashi_total - aoki_current + 1

print(runs_needed)