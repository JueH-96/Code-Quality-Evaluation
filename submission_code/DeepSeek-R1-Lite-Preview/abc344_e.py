def main():
    import sys
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr+N]))
    ptr += N
    Q = int(data[ptr])
    ptr += 1
    queries = data[ptr:]
    
    # Build prev and next dictionaries
    prev = {}
    next = {}
    # Initialize head and tail
    head = A[0]
    tail = A[-1]
    for i in range(N):
        if i > 0:
            prev[A[i]] = A[i-1]
            next[A[i-1]] = A[i]
        if i < N-1:
            next[A[i]] = A[i+1]
            prev[A[i+1]] = A[i]
    
    # Process queries
    q_ptr = 0
    for _ in range(Q):
        query_type = queries[q_ptr]
        if query_type == '1':
            x = int(queries[q_ptr+1])
            y = int(queries[q_ptr+2])
            # Insert y after x
            next_x = next.get(x, None)
            next[x] = y
            next[y] = next_x
            prev[y] = x
            if next_x is not None:
                prev[next_x] = y
            if next[x] == None:
                tail = y
            q_ptr += 3
        elif query_type == '2':
            x = int(queries[q_ptr+1])
            # Remove x
            prev_x = prev.get(x, None)
            next_x = next.get(x, None)
            if prev_x is not None:
                next[prev_x] = next_x
            else:
                head = next_x
            if next_x is not None:
                prev[next_x] = prev_x
            else:
                tail = prev_x
            del prev[x]
            del next[x]
            q_ptr += 2
    
    # Reconstruct the sequence starting from head
    result = []
    current = head
    while current is not None:
        result.append(str(current))
        current = next.get(current, None)
    
    # Print the result
    print(' '.join(result))

if __name__ == '__main__':
    main()