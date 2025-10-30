def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, L, R = map(int, data)
    
    A = list(range(1, N+1))
    
    # Reverse the sublist from L to R (1-based → 0-based indices)
    A[L-1:R] = reversed(A[L-1:R])
    
    # Print the resulting sequence
    print(' '.join(map(str, A)))

# Do not forget to call main() at the end
main()