# YOUR CODE HERE
def max_subsidy_limit(N, M, A):
    def total_subsidy(x):
        return sum(min(x, a) for a in A)
    
    left, right = 0, max(A)
    
    if total_subsidy(right) <= M:
        return "infinite"
    
    while left < right:
        mid = (left + right + 1) // 2
        if total_subsidy(mid) <= M:
            left = mid
        else:
            right = mid - 1
    
    return left

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

print(max_subsidy_limit(N, M, A))