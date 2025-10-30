def solve():
    # Parse input: N (sequence length), K (number of elements to sum), Q (number of updates)
    N, K, Q = map(int, input().split())
    
    # Initialize sequence with all zeros
    sequence = [0] * N
    
    # Process each update
    for _ in range(Q):
        X, Y = map(int, input().split())
        
        # Update the sequence at position X (converting to 0-indexed)
        sequence[X-1] = Y
        
        # Calculate f(A): sum of K largest elements
        # Sort the sequence in descending order
        sorted_seq = sorted(sequence, reverse=True)
        
        # Sum the first K elements
        result = sum(sorted_seq[:K])
        
        # Output the result
        print(result)

solve()