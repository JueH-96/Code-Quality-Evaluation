def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Calculate B_i = A_i * A_{i+1} for i in [1..N-1]
    B = [A[i] * A[i+1] for i in range(N - 1)]
    
    # Print the resulting values separated by spaces
    print(" ".join(map(str, B)))

# Do not forget to call main(), or you will not be awarded any points!
main()