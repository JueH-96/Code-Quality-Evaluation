def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # Parse input
    N = int(input_data[0])
    K = int(input_data[1])
    A = list(map(int, input_data[2:2+N]))
    
    # Use set to remove duplicates and then sort it.
    unique_A = sorted(set(A))
    
    # Helper function for arithmetic series sum: sum from low to high inclusive.
    def series_sum(low, high):
        # Number of terms
        n = high - low + 1
        return (low + high) * n // 2
    
    total = 0
    start = 1
    
    # Iterate over unique sorted values in A.
    for a in unique_A:
        if a > K:
            # If a is greater than K, then we only need missing numbers from start to K.
            break
        if a > start:
            # Segment from start to a-1 are missing.
            total += series_sum(start, a - 1)
        # Update start to a+1 since a is present.
        start = a + 1
        # If start is already beyond K, break early.
        if start > K:
            break
            
    # After processing, if there remain numbers from start to K, add them.
    if start <= K:
        total += series_sum(start, K)
    
    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()