def main():
    import sys
    input_data = sys.stdin.read().split()
    # First value is N, then the list of integers
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # Create B list by multiplying adjacent elements
    B = [A[i] * A[i+1] for i in range(N-1)]
    
    # Print the results space-separated
    print(" ".join(map(str, B)))
    
if __name__ == '__main__':
    main()