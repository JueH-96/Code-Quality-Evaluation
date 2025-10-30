def process_queries():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    A = list(map(int, data[1].split()))
    Q = int(data[2])
    
    queries = data[3:3 + Q]
    
    # Create a mapping from value to its index for quick access
    index_map = {value: idx for idx, value in enumerate(A)}
    
    for query in queries:
        parts = list(map(int, query.split()))
        if parts[0] == 1:  # Insert y after x
            x, y = parts[1], parts[2]
            idx = index_map[x] + 1
            A.insert(idx, y)
            # Update the index map
            index_map[y] = idx
            # Update indices for all elements after the insertion point
            for i in range(idx + 1, len(A)):
                index_map[A[i]] = i
        elif parts[0] == 2:  # Remove x
            x = parts[1]
            idx = index_map[x]
            A.pop(idx)
            # Remove from index map
            del index_map[x]
            # Update indices for all elements after the removal point
            for i in range(idx, len(A)):
                index_map[A[i]] = i
    
    print(" ".join(map(str, A)))