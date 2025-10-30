def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    T = list(map(int, data[3:3+N]))
    
    # Initialize the earliest possible shipping day
    earliest_day = 0
    total_dissatisfaction = 0
    
    i = 0
    while i < N:
        # Determine the earliest possible shipping day for the current batch
        current_day = max(earliest_day, T[i])
        
        # Determine how many orders can be shipped in this batch
        # We can ship up to K orders, but no more than the remaining orders
        batch_size = min(K, N - i)
        
        # Calculate the dissatisfaction for this batch
        # All orders in the batch are shipped on current_day
        for j in range(i, i + batch_size):
            total_dissatisfaction += max(0, current_day - T[j])
        
        # Update the earliest_day for the next batch
        earliest_day = current_day + X
        
        # Move to the next batch
        i += batch_size
    
    print(total_dissatisfaction)

if __name__ == "__main__":
    main()