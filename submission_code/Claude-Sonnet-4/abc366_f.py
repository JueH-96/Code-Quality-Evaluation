from itertools import permutations

def solve():
    N, K = map(int, input().split())
    functions = []
    for _ in range(N):
        A, B = map(int, input().split())
        functions.append((A, B))
    
    max_value = 0
    
    # Try all permutations of K functions from N functions
    for perm in permutations(range(N), K):
        # Start with x = 1
        x = 1
        
        # Apply functions in sequence
        for i in perm:
            A, B = functions[i]
            x = A * x + B
        
        max_value = max(max_value, x)
    
    print(max_value)

solve()