# YOUR CODE HERE
def solve(N, M, A):
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % M
    
    count = [0] * M
    result = 0
    
    for r in range(N + 1):
        result += r * prefix_sum[r]
        result -= sum(count[k] * k for k in range(M))
        count[prefix_sum[r]] += 1
    
    return result

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Solve and print result
print(solve(N, M, A))