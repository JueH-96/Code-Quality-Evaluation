def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M, P = map(int, data)
    
    if M > N:
        print(0)
        return
    
    # The first full moon day is M.
    # Subsequent ones occur in intervals of P.
    # Count how many such days are <= N.
    # This is given by 1 + ((N - M) // P).
    
    answer = 1 + (N - M) // P
    print(answer)

solve()