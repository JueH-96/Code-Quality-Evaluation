import sys

def solve_test_case():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Sort A in descending order
    A.sort(reverse=True)
    
    # Find the minimum value of the expression
    min_value = float('inf')
    for i in range(N-K+1):
        max_A = A[i]
        sum_B = sum(B[i:i+K])
        min_value = min(min_value, max_A * sum_B)
    
    return min_value

T = int(input())
for _ in range(T):
    print(solve_test_case())