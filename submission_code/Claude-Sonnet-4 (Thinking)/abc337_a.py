# Read number of matches
N = int(input())

# Initialize total scores
takahashi_total = 0
aoki_total = 0

# Process each match
for i in range(N):
    x, y = map(int, input().split())
    takahashi_total += x
    aoki_total += y

# Determine the winner
if takahashi_total > aoki_total:
    print("Takahashi")
elif aoki_total > takahashi_total:
    print("Aoki")
else:
    print("Draw")