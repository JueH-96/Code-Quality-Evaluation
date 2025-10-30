def main():
    import sys
    input_data = sys.stdin.read().split()
    it = iter(input_data)
    
    n = int(next(it))
    # Read initial sequence
    seq = [int(next(it)) for _ in range(n)]
    
    # Dictionaries for prev and next pointers.
    # For each element x, prev[x] gives previous element in seq (or None if x is head)
    # Similarly, next[x] gives next element (or None if x is tail)
    prev = {}
    nxt = {}
    
    # Build doubly-linked setup:
    for i in range(len(seq)):
        if i == 0:
            prev[seq[i]] = None
        else:
            prev[seq[i]] = seq[i-1]
        if i == len(seq) - 1:
            nxt[seq[i]] = None
        else:
            nxt[seq[i]] = seq[i+1]
    
    # Head pointer to track beginning of the list.
    head = seq[0]
    
    q = int(next(it))
    for _ in range(q):
        typ = next(it)
        if typ == '1':
            # Insert operation: "1 x y" means insert y immediately after x
            x = int(next(it))
            y = int(next(it))
            # New node y: its previous is x, and its next is what used to follow x.
            nxt_y = nxt[x]
            nxt[y] = nxt_y
            prev[y] = x
            nxt[x] = y
            # If there was an element following x, update its prev pointer.
            if nxt_y is not None:
                prev[nxt_y] = y
        elif typ == '2':
            # Deletion operation: "2 x" means remove x from the list
            x = int(next(it))
            left = prev[x]
            right = nxt[x]
            if left is not None:
                nxt[left] = right
            else:
                # if left is None, then x was head
                head = right
            if right is not None:
                prev[right] = left
            # Optionally remove x from dictionaries (not strictly necessary, but safe)
            # We can remove them if memory is a concern.
            #del prev[x]
            #del nxt[x]
    
    # After processing queries, traverse the list from the head and print the answer.
    result = []
    cur = head
    while cur is not None:
        result.append(str(cur))
        cur = nxt[cur]
    sys.stdout.write(" ".join(result))
    
if __name__ == '__main__':
    main()