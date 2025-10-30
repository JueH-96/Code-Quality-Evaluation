def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Read N and the initial sequence A
    N = int(input_data[0])
    A = list(map(int, input_data[1:1+N]))
    
    # Number of queries
    Q = int(input_data[1+N])
    
    # We'll keep dictionaries for next and prev to simulate a doubly-linked list
    next_elem = {}
    prev_elem = {}
    
    # Initialize the linked structure for the initial sequence
    head = A[0]
    tail = A[-1]
    
    prev_elem[head] = None
    for i in range(N - 1):
        next_elem[A[i]] = A[i + 1]
        prev_elem[A[i + 1]] = A[i]
    next_elem[tail] = None
    
    # Index to read queries
    idx = 2 + N  # start reading queries from here
    
    # Process each query
    for _ in range(Q):
        t = int(input_data[idx])
        idx += 1
        
        if t == 1:
            # 1 x y : Insert y immediately after x
            x = int(input_data[idx]); y = int(input_data[idx+1])
            idx += 2
            
            # next_elem[x] -> inserted y -> right
            right = next_elem.get(x, None)
            
            next_elem[x] = y
            prev_elem[y] = x
            if right is not None:
                next_elem[y] = right
                prev_elem[right] = y
            else:
                next_elem[y] = None
                tail = y
            
        else:
            # 2 x : Remove x from A
            x = int(input_data[idx])
            idx += 1
            
            left = prev_elem.get(x, None)
            right = next_elem.get(x, None)
            
            if left is not None:
                next_elem[left] = right
            else:
                head = right
            
            if right is not None:
                prev_elem[right] = left
            else:
                tail = left
            
            # Cleanup (not strictly necessary, but can omit if desired)
            if x in next_elem:
                del next_elem[x]
            if x in prev_elem:
                del prev_elem[x]
    
    # Traverse from head to reconstruct the sequence
    result = []
    curr = head
    while curr is not None:
        result.append(curr)
        curr = next_elem.get(curr, None)
    
    print(" ".join(map(str, result)))


def main():
    solve()

if __name__ == "__main__":
    main()