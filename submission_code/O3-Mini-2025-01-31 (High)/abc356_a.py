def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N, L, R = map(int, data)
    # Create the sequence A = (1, 2, ..., N)
    A = list(range(1, N + 1))
    # Reverse the L-th through R-th elements (adjusting for 0-index)
    A[L - 1:R] = A[L - 1:R][::-1]
    # Print the resulting sequence with spaces
    print(" ".join(map(str, A)))

if __name__ == '__main__':
    main()