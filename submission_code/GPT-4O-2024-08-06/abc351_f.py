# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Initialize the result
    result = 0
    
    # Iterate over each i
    for i in range(N):
        # For each i, iterate over each j > i
        for j in range(i + 1, N):
            # Calculate max(A[j] - A[i], 0) and add to result
            result += max(A[j] - A[i], 0)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()