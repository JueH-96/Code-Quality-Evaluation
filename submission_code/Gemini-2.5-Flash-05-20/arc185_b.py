import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    total_sum = sum(A)
    
    current_prefix_sum = 0
    
    # Iterate through the array (0-indexed)
    for i in range(N):
        current_prefix_sum += A[i]
        
        # The condition to check is P_k <= k * (S / N)
        # For 0-indexed array, P_{i+1} is current_prefix_sum, and k is (i+1).
        # So, current_prefix_sum <= (i+1) * (total_sum / N)
        # To avoid floating point issues, we cross-multiply:
        # current_prefix_sum * N <= (i+1) * total_sum
        # If this condition is violated, it's impossible.
        if current_prefix_sum * N > (i + 1) * total_sum:
            print("No")
            return
            
    # If the loop completes, it means the condition held for all prefixes.
    print("Yes")

T = int(sys.stdin.readline())
for _ in range(T):
    solve()