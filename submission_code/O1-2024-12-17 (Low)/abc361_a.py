def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, K, X = map(int, data[:3])
    A = list(map(int, data[3:]))

    # Insert X after the K-th element (index K-1)
    B = A[:K] + [X] + A[K:]

    print(" ".join(map(str, B)))

# Do not forget to call the main function
main()