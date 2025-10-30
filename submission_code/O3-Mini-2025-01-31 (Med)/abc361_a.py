def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    K = int(input_data[1])
    X = int(input_data[2])
    A = list(map(int, input_data[3:]))
    
    # Insert X immediately after the K-th element (K is 1-indexed)
    B = A[:K] + [X] + A[K:]
    
    print(" ".join(map(str, B)))

if __name__ == '__main__':
    main()