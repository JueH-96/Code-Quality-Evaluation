# Read the input
N = int(input())
scores = []
for _ in range(N):
    x, y = map(int, input().split())
    scores.append((x, y))

# Calculate the total scores for each team
takahashi_total = sum(x for x, y in scores)
aoki_total = sum(y for x, y in scores)

# Determine the winner
if takahashi_total > aoki_total:
    print("Takahashi")
elif aoki_total > takahashi_total:
    print("Aoki")
else:
    print("Draw")