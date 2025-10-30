import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1

    H = list(map(int, input[idx:idx+N]))
    idx += N
    H = [0] + H  # 1-based indexing

    # Compute prev array
    prev = [0] * (N + 1)  # prev[1..N]
    stack = []
    for j in range(1, N + 1):
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        if stack:
            prev[j] = stack[-1]
        else:
            prev[j] = 0
        stack.append(j)

    # Build the segment tree
    data = prev[1:N+1]  # 0-based indices 0..N-1
    n = len(data)
    size = 1
    while size < n:
        size <<= 1
    tree = [[] for _ in range(2 * size)]

    # Fill leaves
    for i in range(n):
        tree[size + i] = [data[i]]
    # Fill the rest
    for i in range(size - 1, 0, -1):
        left = tree[2 * i]
        right = tree[2 * i + 1]
        merged = []
        a, b = 0, 0
        while a < len(left) and b < len(right):
            if left[a] < right[b]:
                merged.append(left[a])
                a += 1
            else:
                merged.append(right[b])
                b += 1
        merged.extend(left[a:])
        merged.extend(right[b:])
        tree[i] = merged

    # Process queries
    output = []
    for _ in range(Q):
        l = int(input[idx])
        idx += 1
        r = int(input[idx])
        idx += 1
        A = max(r + 1, l + 1)
        if A > N:
            output.append('0')
            continue
        a = A - 1
        b = N - 1
        res = 0
        l_query = l
        l_node = a + size
        r_node = b + size
        while l_node <= r_node:
            if l_node % 2 == 1:
                res += bisect.bisect_left(tree[l_node], l_query)
                l_node += 1
            if r_node % 2 == 0:
                res += bisect.bisect_left(tree[r_node], l_query)
                r_node -= 1
            l_node >>= 1
            r_node >>= 1
        output.append(str(res))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()