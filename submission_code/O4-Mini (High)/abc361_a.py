def main():
    # Read N (length of A), K (position), X (value to insert)
    N, K, X = map(int, input().split())
    # Read the sequence A
    A = list(map(int, input().split()))
    # Build the new sequence by inserting X after the K-th element (1-indexed)
    B = A[:K] + [X] + A[K:]
    # Print the result
    print(*B)

if __name__ == "__main__":
    main()