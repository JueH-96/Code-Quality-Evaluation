from itertools import combinations

def solve():
    # Read input
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Generate all possible combinations of K elements
    max_xor = 0
    for comb in combinations(A, K):
        # Calculate XOR of current combination
        current_xor = 0
        for num in comb:
            current_xor ^= num
        max_xor = max(max_xor, current_xor)
    
    print(max_xor)

# Run the solution
solve()