def can_have_fixed_elements(N, K):
    # The maximum number of fixed elements is N^2
    max_fixed = N * N
    
    # The minimum number of fixed elements is 0
    min_fixed = 0
    
    # The number of fixed elements must be such that:
    # 1. It can be achieved by some arrangement of 0s and 1s
    # 2. The number of fixed elements must be consistent with the row and column sums
    
    # The number of fixed elements must be even if N is even
    # The number of fixed elements can be odd if N is odd
    if N % 2 == 0:
        if K % 2 != 0:
            return False
    return min_fixed <= K <= max_fixed

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    K_values = list(map(int, data[2:2 + Q]))
    
    results = []
    for K in K_values:
        if can_have_fixed_elements(N, K):
            results.append("Yes")
        else:
            results.append("No")
    
    print("
".join(results))