# YOUR CODE HERE
def solve(N, S, A):
    # Create a list to store cumulative sums
    cumsum = [0] * (2*N + 1)
    for i in range(2*N):
        cumsum[i+1] = cumsum[i] + A[i % N]
    
    # Use two pointers to check for a subsequence with sum S
    left, right = 0, 1
    while right <= 2*N and left < right:
        current_sum = cumsum[right] - cumsum[left]
        if current_sum == S:
            return "Yes"
        elif current_sum < S:
            right += 1
        else:
            left += 1
    
    return "No"

# Read input
N, S = map(int, input().split())
A = list(map(int, input().split()))

# Solve and print output
print(solve(N, S, A))