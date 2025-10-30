def solve(N):
    # For N = 1, any (A, M) where gcd(A, M) = 1 works
    if N == 1:
        return 20250126, 1
    
    # For other N, we can use a simple construction:
    # Choose M = kN + 1 for some k, and find A with order N
    
    # Try small values first
    for k in range(1, 10000):
        M = k * N + 1
        
        # Try A = 2 first
        A = 2
        if pow(A, N, M) == 1:
            # Check if N is the minimal order
            is_minimal = True
            for d in range(1, N):
                if N % d == 0 and pow(A, d, M) == 1:
                    is_minimal = False
                    break
            if is_minimal:
                return A, M
        
        # Try other small values of A
        for A in range(3, min(100, M)):
            if pow(A, N, M) == 1:
                # Check if N is the minimal order
                is_minimal = True
                for d in range(1, N):
                    if N % d == 0 and pow(A, d, M) == 1:
                        is_minimal = False
                        break
                if is_minimal:
                    return A, M
    
    # This should not happen for valid inputs
    return 2, 2 * N + 1

# Read input
T = int(input())
for _ in range(T):
    N = int(input())
    A, M = solve(N)
    print(A, M)