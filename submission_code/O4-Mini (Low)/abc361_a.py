def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    A = list(map(int, data[3:]))

    # Insert X immediately after the K-th element (1-indexed)
    # In Python's 0-indexed list, that position is K.
    B = A[:K] + [X] + A[K:]
    
    print(*B)

if __name__ == "__main__":
    main()