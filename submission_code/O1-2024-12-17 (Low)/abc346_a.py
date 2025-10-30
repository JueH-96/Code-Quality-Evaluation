def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Compute products
    B = [A[i] * A[i+1] for i in range(N-1)]
    
    # Print the results separated by space
    print(" ".join(map(str, B)))

# Do not forget to call main
main()