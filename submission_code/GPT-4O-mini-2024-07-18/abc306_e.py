def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N, K, Q
    N, K, Q = map(int, data[0].split())
    
    # Initialize the sequence A
    A = [0] * N
    
    # Prepare to collect results
    results = []
    
    # Function to calculate f(A)
    def calculate_f(A):
        # Sort A in descending order and sum the top K elements
        return sum(sorted(A, reverse=True)[:K])
    
    # Process each update
    for i in range(1, Q + 1):
        X, Y = map(int, data[i].split())
        # Update A[X-1] to Y (convert 1-based index to 0-based)
        A[X - 1] = Y
        # Calculate f(A) and store the result
        results.append(calculate_f(A))
    
    # Print all results
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()