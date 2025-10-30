def main():
    import sys
    data = sys.stdin.read().split()
    idx = 0

    # Read the initial sequence length and the sequence itself.
    n = int(data[idx])
    idx += 1
    arr = [int(x) for x in data[idx:idx+n]]
    idx += n

    # Build a doubly-linked list using a dictionary.
    # For each element, we store a pair [prev, next]
    ll = {}
    for i, val in enumerate(arr):
        prev_val = arr[i-1] if i > 0 else None
        next_val = arr[i+1] if i < n-1 else None
        ll[val] = [prev_val, next_val]
    head = arr[0]
    tail = arr[-1]

    # Process Q queries
    q = int(data[idx])
    idx += 1
    for _ in range(q):
        typ = data[idx]
        idx += 1
        if typ == '1':
            # Query "1 x y" means insert y immediately after x.
            x = int(data[idx])
            y = int(data[idx+1])
            idx += 2

            nxt = ll[x][1]  # Get the node after x.
            ll[y] = [x, nxt]  # New node y has x as previous and nxt as next.
            ll[x][1] = y      # Update x's next to y.
            if nxt is not None:
                ll[nxt][0] = y  # Update nxt's previous to y.
            else:
                tail = y  # y becomes the new tail.
        else:
            # Query "2 x" means remove the element x from the list.
            x = int(data[idx])
            idx += 1

            left, right = ll[x]
            if left is not None:
                ll[left][1] = right
            else:
                head = right  # x was the head.
            if right is not None:
                ll[right][0] = left
            else:
                tail = left  # x was the tail.
            del ll[x]  # Remove x from the dictionary.

    # Traverse the list starting from head and print the result.
    res = []
    cur = head
    while cur is not None:
        res.append(str(cur))
        cur = ll[cur][1]  # Move to the next element.
    sys.stdout.write(" ".join(res))

if __name__ == '__main__':
    main()