# Read the number of matches
N = int(input())

# Initialize scores for both teams
takahashi_score = 0
aoki_score = 0

# Read scores for each match and update the total scores
for _ in range(N):
    x, y = map(int, input().split())
    takahashi_score += x
    aoki_score += y

# Determine the winner
if takahashi_score > aoki_score:
    print("Takahashi")
elif takahashi_score < aoki_score:
    print("Aoki")
else:
    print("Draw")