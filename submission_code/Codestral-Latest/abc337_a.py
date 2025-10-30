# Read the number of matches
N = int(input().strip())

# Initialize the total scores for both teams
takahashi_score = 0
aoki_score = 0

# Read the scores for each match and update the total scores
for _ in range(N):
    X, Y = map(int, input().strip().split())
    takahashi_score += X
    aoki_score += Y

# Determine the winner
if takahashi_score > aoki_score:
    print("Takahashi")
elif aoki_score > takahashi_score:
    print("Aoki")
else:
    print("Draw")