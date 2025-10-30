# YOUR CODE HERE
N, T, A = map(int, input().split())

# Since N is odd, need more than N/2 to win
# Which means need at least (N+1)/2 votes to win
majority_needed = (N + 1) // 2

# Check if either candidate has already secured enough votes
# or if either candidate cannot possibly win
if T >= majority_needed or A >= majority_needed:
    print("Yes")
else:
    print("No")