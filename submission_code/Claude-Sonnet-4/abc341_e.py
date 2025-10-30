# YOUR CODE HERE
def solve():
    N, Q = map(int, input().split())
    S = list(input().strip())
    
    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:  # Flip query
            L, R = query[1], query[2]
            for i in range(L-1, R):  # Convert to 0-indexed
                S[i] = '1' if S[i] == '0' else '0'
        
        else:  # Check query
            L, R = query[1], query[2]
            # Extract substring S[L-1:R] (convert to 0-indexed)
            substring = S[L-1:R]
            
            # Check if it's a good string
            is_good = True
            for i in range(len(substring) - 1):
                if substring[i] == substring[i + 1]:
                    is_good = False
                    break
            
            print("Yes" if is_good else "No")

solve()