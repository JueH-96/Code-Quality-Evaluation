# Read the number of matches
N = int(input())

# Initialize total scores for both teams
takahashi_total = 0
aoki_total = 0

# Process each match
for i in range(N):
    # Read scores for current match
    X, Y = map(int, input().split())
    
    # Update totals
    takahashi_total += X
    aoki_total += Y

# Determine and output the winner
if takahashi_total > aoki_total:
    print("Takahashi")
elif aoki_total > takahashi_total:
    print("Aoki")
else:
    print("Draw")