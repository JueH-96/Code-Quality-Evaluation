def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    X = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    # Insert X immediately after the K-th element (1-based):
    # In 0-based indexing, that's at position K.
    A.insert(K, X)
    # Print the resulting sequence
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()