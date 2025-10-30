def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    queries = data[3:]
    
    # Compute D[i] = S[i] XOR S[i+1] for i from 0 to N-2
    D = []
    for i in range(N-1):
        D.append(int(S[i]) ^ int(S[i+1]))
    
    # Segment tree implementation
    size = len(D)
    tree = [0] * (2 * size)
    for i in range(size):
        tree[size + i] = 1 - D[i]
    for i in range(size - 1, 0, -1):
        tree[i] = tree[i << 1] + tree[i << 1 | 1]
    
    # Lazy propagation for range XOR updates
    lazy = [0] * (2 * size)
    
    def push(node, node_left, node_right):
        if lazy[node]:
            tree[node] = (node_right - node_left + 1) - tree[node]
            if node < size:
                lazy[node << 1] ^= 1
                lazy[node << 1 | 1] ^= 1
            lazy[node] = 0
    
    def range_xor_update(a, b):
        a += size
        b += size
        la, ra = a, b
        while a <= b:
            if a % 2 == 1:
                push(a >> 1, (a >> 1) << 1 - size, (a >> 1) << 1 | 1 - size)
                tree[a] ^= 1
                a += 1
            if b % 2 == 0:
                push(b >> 1, (b >> 1) << 1 - size, (b >> 1) << 1 | 1 - size)
                tree[b] ^= 1
                b -= 1
            a >>= 1
            b >>= 1
        a, b = la, ra
        while a > 1:
            a >>= 1
            tree[a] = tree[a << 1] + tree[a << 1 | 1]
            if lazy[a]:
                tree[a] = ( (a << 1 | 1) - (a << 1) + 1 ) - tree[a]
        while b > 1:
            b >>= 1
            tree[b] = tree[b << 1] + tree[b << 1 | 1]
            if lazy[b]:
                tree[b] = ( (b << 1 | 1) - (b << 1) + 1 ) - tree[b]
    
    def range_sum_query(a, b):
        a += size
        b += size
        res = 0
        while a <= b:
            if a % 2 == 1:
                push(a >> 1, (a >> 1) << 1 - size, (a >> 1) << 1 | 1 - size)
                res += tree[a]
                a += 1
            if b % 2 == 0:
                push(b >> 1, (b >> 1) << 1 - size, (b >> 1) << 1 | 1 - size)
                res += tree[b]
                b -= 1
            a >>= 1
            b >>= 1
        return res
    
    idx = 0
    output = []
    for _ in range(Q):
        type_q = queries[idx]
        L = int(queries[idx + 1]) - 1  # 0-based indexing
        R = int(queries[idx + 2]) - 1  # 0-based indexing
        idx += 3
        if type_q == '1':
            # Flip operation: affects D[L-1] to D[R] if possible
            if L > 0:
                range_xor_update(L - 1, min(R, size - 1))
            if R < size:
                range_xor_update(L, min(R, size - 1))
        elif type_q == '2':
            if L == R:
                output.append("Yes")
            else:
                sum_bad = range_sum_query(L, R - 1)
                if sum_bad == 0:
                    output.append("Yes")
                else:
                    output.append("No")
    print("
".join(output))

if __name__ == "__main__":
    main()