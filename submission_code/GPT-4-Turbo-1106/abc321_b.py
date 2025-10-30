# Read input values
N, X = map(int, input().split())
A = list(map(int, input().split()))

# Sort the scores to easily find the minimum and maximum
A.sort()

# Calculate the sum of the scores excluding the highest and lowest
current_sum = sum(A[1:-1]) if N > 3 else sum(A)

# Calculate the minimum score needed in the last round
min_score_needed = X - current_sum

# Check if the minimum score needed is within the valid range
if min_score_needed > 100:
    print(-1)
elif min_score_needed < 0:
    print(0)
else:
    print(min_score_needed)