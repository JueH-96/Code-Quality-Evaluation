# Read Team Takahashi's scores
takahashi = list(map(int, input().split()))

# Read Team Aoki's scores
aoki = list(map(int, input().split()))

# Calculate total scores
takahashi_total = sum(takahashi)
aoki_total = sum(aoki)

# Aoki needs to score at least 1 more run than Takahashi's total to win
needed = takahashi_total - aoki_total + 1

print(needed)