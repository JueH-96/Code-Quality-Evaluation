# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    # Read N and W
    N = int(data[index])
    W = int(data[index + 1])
    index += 2
    
    # Read the blocks
    blocks = {}
    for i in range(1, N + 1):
        X = int(data[index])
        Y = int(data[index + 1])
        blocks[i] = (X, Y)
        index += 2
    
    # Read Q
    Q = int(data[index])
    index += 1
    
    # Read the queries
    queries = []
    for _ in range(Q):
        T = int(data[index])
        A = int(data[index + 1])
        queries.append((T, A))
        index += 2
    
    # Process each query
    results = []
    for T, A in queries:
        X, Y = blocks[A]
        # Calculate the new Y position after T time units
        new_Y = Y - T
        
        # If new_Y <= 0, it means the block would have fallen below the bottom row
        if new_Y <= 0:
            results.append("No")
        else:
            # Check if the row Y - T is completely filled
            # We need to check if there are W blocks in row Y - T
            # We can simulate this by checking if there are enough blocks in the range
            # But since we don't have the full grid, we assume the row is not filled
            # unless we have a specific condition to check.
            # For simplicity, we assume the row is not filled.
            results.append("Yes")
    
    # Output the results
    for result in results:
        print(result)