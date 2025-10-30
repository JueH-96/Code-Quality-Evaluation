def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    
    # Read initial list size and the list elements
    n = int(next(it))
    arr = [int(next(it)) for _ in range(n)]
    
    # Build a doubly linked list using a dictionary.
    # For each element, store [prev, next] pointers.
    linked = {}
    for i in range(n):
        prev_elem = arr[i-1] if i > 0 else None
        next_elem = arr[i+1] if i < n-1 else None
        linked[arr[i]] = [prev_elem, next_elem]
    head = arr[0]
    
    # Process queries
    q = int(next(it))
    for _ in range(q):
        typ = next(it)
        if typ == '1':
            # Query format: 1 x y (insert y immediately after x)
            x = int(next(it))
            y = int(next(it))
            nxt = linked[x][1]  # The element after x
            # Create a new node for y
            linked[y] = [x, nxt]
            # Update x's next pointer to y
            linked[x][1] = y
            # If there was a next node, update its previous pointer to y
            if nxt is not None:
                linked[nxt][0] = y
        else:
            # Query format: 2 x (remove element x)
            x = int(next(it))
            prev_x, next_x = linked[x]
            # If x is the head, update head pointer
            if prev_x is None:
                head = next_x
            else:
                linked[prev_x][1] = next_x
            if next_x is not None:
                linked[next_x][0] = prev_x
            # Remove x from the dictionary
            del linked[x]
    
    # Traverse the list starting from head to build the result
    output = []
    cur = head
    while cur is not None:
        output.append(str(cur))
        cur = linked[cur][1]
    
    sys.stdout.write(" ".join(output))


if __name__ == '__main__':
    main()