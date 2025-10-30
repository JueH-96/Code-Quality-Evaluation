def main():
    import sys
    input_data = sys.stdin.read().split()
    # The first element is N, followed by A_1, A_2, ..., A_N
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # Compute B_i = A_i * A_{i+1} for i = 0 to N-2 and collect results
    result = []
    for i in range(N - 1):
        result.append(A[i] * A[i+1])
    
    # Print results separated by spaces
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()