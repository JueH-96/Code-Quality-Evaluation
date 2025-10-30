def max_subsidy_limit(N, M, A):
    def can_subsidize(x):
        total_subsidy = sum(min(x, a) for a in A)
        return total_subsidy <= M

    left, right = 0, max(A)
    
    if can_subsidize(right):
        return "infinite"
    
    while left < right:
        mid = (left + right + 1) // 2
        if can_subsidize(mid):
            left = mid
        else:
            right = mid - 1
    
    return left

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Output the result
print(max_subsidy_limit(N, M, A))