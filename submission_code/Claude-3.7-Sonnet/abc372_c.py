# YOUR CODE HERE
def solve():
    N, Q = map(int, input().split())
    S = list(input().strip())  # Convert to list for easy modification
    
    def is_ABC(pos):
        return 0 <= pos < len(S) - 2 and S[pos] == 'A' and S[pos+1] == 'B' and S[pos+2] == 'C'
    
    # Count initial occurrences of "ABC"
    count = sum(1 for i in range(len(S) - 2) if is_ABC(i))
    
    for _ in range(Q):
        X, C = input().split()
        X = int(X) - 1  # Convert to 0-based indexing
        
        # Check if positions affected by replacement had "ABC"
        for pos in [X-2, X-1, X]:
            if is_ABC(pos):
                count -= 1
        
        # Perform replacement
        S[X] = C
        
        # Check if positions after replacement have "ABC"
        for pos in [X-2, X-1, X]:
            if is_ABC(pos):
                count += 1
        
        print(count)

solve()