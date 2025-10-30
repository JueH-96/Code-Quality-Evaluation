import sys

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr+N]))
    ptr += N

    # Initialize count and missing arrays
    max_val = N
    count = [0] * (N + 2)  # indices 0 to N+1
    for x in A:
        if x <= N:
            count[x] += 1

    missing = [False] * (N + 1 + 1)  # indices 0 to N
    for x in range(N + 1):
        missing[x] = (count[x] == 0)

    # Build segment tree
    INF = N + 2
    L = N + 1
    size = 1
    while size < L:
        size <<= 1
    st = [INF] * (2 * size)

    # Initialize leaves
    for x in range(L):
        if x <= N:
            if missing[x]:
                st[size + x] = x
            else:
                st[size + x] = INF
        else:
            st[size + x] = INF

    # Fill padding leaves beyond L
    for x in range(L, size):
        st[size + x] = INF

    # Build non-leaf nodes
    for i in range(size - 1, 0, -1):
        st[i] = min(st[2 * i], st[2 * i + 1])

    # Update function
    def update_seg(x):
        pos = size + x
        if missing[x]:
            st[pos] = x
        else:
            st[pos] = INF
        pos >>= 1
        while pos >= 1:
            new_val = min(st[2 * pos], st[2 * pos + 1])
            if st[pos] == new_val:
                break
            st[pos] = new_val
            pos >>= 1

    output = []
    for _ in range(Q):
        i_k = int(data[ptr]) - 1  # convert to 0-based index
        ptr += 1
        x_k = int(data[ptr])
        ptr += 1

        old_val = A[i_k]
        new_val = x_k

        # Process old_val
        if old_val <= N:
            count[old_val] -= 1
            new_missing = (count[old_val] == 0)
            if missing[old_val] != new_missing:
                missing[old_val] = new_missing
                update_seg(old_val)

        # Process new_val
        if new_val <= N:
            old_count = count[new_val]
            count[new_val] += 1
            new_missing = (count[new_val] == 0)
            if missing[new_val] != new_missing:
                missing[new_val] = new_missing
                update_seg(new_val)

        # Update array
        A[i_k] = new_val

        # mex is the root of the segment tree
        mex = st[1]
        output.append(str(mex))

    print('
'.join(output))

if __name__ == "__main__":
    main()