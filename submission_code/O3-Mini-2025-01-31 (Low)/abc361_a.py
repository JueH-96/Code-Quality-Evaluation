def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    K = int(input_data[1])
    X = int(input_data[2])
    A = list(map(int, input_data[3:]))
    # Insert X immediately after the K-th element (0-indexed: after index K-1)
    A.insert(K, X)
    # Print out the updated list as required
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()