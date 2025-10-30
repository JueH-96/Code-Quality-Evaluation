def main():
    import sys
    data = sys.stdin.read().split()
    
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    Q = int(data[1+N])
    idx = 1 + N + 1
    
    # Dictionaries to store the left and right neighbor of each element
    left = {}
    right = {}
    
    # Initialize the doubly-linked list structure from the initial array A
    left[A[0]] = None
    for i in range(1, N):
        left[A[i]] = A[i-1]
        right[A[i-1]] = A[i]
    right[A[-1]] = None
    
    head = A[0]      # Track the current head of the list
    tail = A[-1]     # Track the current tail of the list
    
    in_list = set(A) # Track which elements are currently in the list
    
    # Process queries
    for _ in range(Q):
        t = int(data[idx]); idx += 1
        if t == 1:
            # 1 x y : Insert y immediately after x
            x = int(data[idx]); idx += 1
            y = int(data[idx]); idx += 1
            
            # Insert y after x
            nxt = right.get(x, None)   # neighbor to the right of x
            right[x] = y
            left[y] = x
            if nxt is not None:
                left[nxt] = y
                right[y] = nxt
            else:
                right[y] = None
                tail = y
            
            in_list.add(y)
        
        else:
            # 2 x : Remove x from A
            x = int(data[idx]); idx += 1
            lft = left[x]
            rgt = right[x]
            
            # If x is head, move head
            if lft is None:
                head = rgt
            else:
                right[lft] = rgt
            
            # If x is tail, move tail
            if rgt is None:
                tail = lft
            else:
                left[rgt] = lft
            
            in_list.remove(x)
    
    # Traverse from head to construct the final list of elements
    result = []
    cur = head
    while cur is not None:
        result.append(cur)
        cur = right.get(cur, None)
    
    print(" ".join(map(str, result)))

# Don't forget to call main()
main()