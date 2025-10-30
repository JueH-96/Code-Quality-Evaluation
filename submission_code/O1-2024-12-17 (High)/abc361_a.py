def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, K, X = map(int, data[:3])
    A = list(map(int, data[3:]))

    # Create the new sequence B by inserting X after the K-th element (1-based index).
    B = A[:K] + [X] + A[K:]

    # Print the result
    print(*B)

# Do not forget to call main()
main()