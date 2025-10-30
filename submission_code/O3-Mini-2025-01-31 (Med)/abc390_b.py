from fractions import Fraction

def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
        
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Any sequence with a single term is trivially geometric,
    # but based on constraints, N will be at least 2.
    if N == 1:
        print("Yes")
        return
    
    # Compute the common ratio as a Fraction
    r = Fraction(A[1], A[0])
    
    # Verify the sequence follows the geometric progression rule:
    # A[i] == A[0] * r^i for all indices i.
    cur = Fraction(A[0])
    is_geometric = True
    for i in range(N):
        if Fraction(A[i]) != cur:
            is_geometric = False
            break
        cur *= r  # multiply by the common ratio for the next term
        
    print("Yes" if is_geometric else "No")
    
if __name__ == '__main__':
    main()