def solve():
    # Read input
    N = int(input())
    Q = list(map(int, input().split()))
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Try all possible combinations of servings
    max_servings = 0
    
    # Find maximum possible servings of dish A based on ingredients
    max_a = float('inf')
    for i in range(N):
        if A[i] > 0:
            max_a = min(max_a, Q[i] // A[i])
    if max_a == float('inf'):
        max_a = 0
        
    # For each possible number of A servings, try B servings
    for a in range(max_a + 1):
        # Calculate remaining ingredients after making a servings of A
        remaining = [Q[i] - a * A[i] for i in range(N)]
        
        # Find maximum possible B servings with remaining ingredients
        max_b = float('inf')
        for i in range(N):
            if B[i] > 0:
                max_b = min(max_b, remaining[i] // B[i])
        if max_b == float('inf'):
            max_b = 0
            
        max_servings = max(max_servings, a + max_b)
    
    print(max_servings)

solve()