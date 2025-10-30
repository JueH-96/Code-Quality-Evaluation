from itertools import permutations

def solve():
    N, K = map(int, input().split())
    
    # Create base sequence with each number appearing K times
    base = []
    for i in range(1, N+1):
        base.extend([i] * K)
        
    # Get all unique permutations
    perms = sorted(set(permutations(base)))
    
    # Get middle sequence
    mid = (len(perms) + 1) // 2 - 1
    
    # Print answer
    print(*perms[mid])

solve()