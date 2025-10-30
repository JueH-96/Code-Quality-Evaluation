# Read the number of matches
N = int(input())

# Initialize total scores
total_takahashi = 0
total_aoki = 0

# Read each match's scores and accumulate totals
for _ in range(N):
    X, Y = map(int, input().split())
    total_takahashi += X
    total_aoki += Y

# Determine and print the result
if total_takahashi > total_aoki:
    print("Takahashi")
elif total_aoki > total_takahashi:
    print("Aoki")
else:
    print("Draw")