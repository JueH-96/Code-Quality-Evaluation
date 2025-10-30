def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    
    try:
        n = int(next(it))
        m = int(next(it))
    except StopIteration:
        return

    # Build an undirected graph where each edge encodes a constraint.
    # For nodes i (0-indexed), graph[i] holds tuples (neighbor, Z) meaning that
    # A[i] xor A[neighbor] must equal Z.
    graph = [[] for _ in range(n)]
    for _ in range(m):
        try:
            x = int(next(it)) - 1
            y = int(next(it)) - 1
            z = int(next(it))
        except StopIteration:
            break
        
        # A self-loop forces A[x] xor A[x] = 0 and must be 0.
        if x == y:
            if z != 0:
                sys.stdout.write("-1")
                return
            # If z==0, nothing to do.
            continue
        
        graph[x].append((y, z))
        graph[y].append((x, z))
    
    # We want to choose a sequence A = (A_1, ..., A_N) with nonnegative integers that
    # minimizes the sum sum(A_i) subject to the constraints.
    # The constraints are given in terms of XOR, which operates bit by bit.
    # Since the sum is sum(A_i) = sum_{bit=0}^{BIT_COUNT-1} ( (1 << bit) * (bit value at that bit) ),
    # we want to set bits to 0 when possible.
    # So for each bit, we decide the value (0 or 1) for each A[i] with the condition that for
    # every edge (u,v) with constraint Z, we have:
    #    (bit_value[u] xor bit_value[v]) == ((Z >> bit) & 1)
    # and we choose free variables to be 0 (which minimizes the sum).
    
    A = [0] * n
    BIT_COUNT = 31  # 2^30 is > 10^9 so 31 bits are sufficient.

    # Process each bit independently.
    for bit in range(BIT_COUNT):
        mask = 1 << bit
        # x_bit will hold the assignment for this bit for each node.
        # -1 means not assigned yet; values will be 0 or 1.
        x_bit = [-1] * n
        
        # Process every connected component.
        for i in range(n):
            if x_bit[i] == -1:
                # For minimization, choose the free variable as 0.
                x_bit[i] = 0
                stack = [i]
                while stack:
                    u = stack.pop()
                    for v, z in graph[u]:
                        # For the current bit, extract the parity constraint from z.
                        # We require x_bit[u] xor x_bit[v] == p, where p is (Z >> bit)&1.
                        parity = 1 if (z & mask) else 0
                        desired = x_bit[u] ^ parity
                        if x_bit[v] == -1:
                            x_bit[v] = desired
                            stack.append(v)
                        elif x_bit[v] != desired:
                            sys.stdout.write("-1")
                            return
        # Incorporate the bit assignment into the final answer A.
        for i in range(n):
            if x_bit[i] == 1:
                A[i] += mask

    # Output the minimal good sequence.
    sys.stdout.write(" ".join(map(str, A)))
    
if __name__ == '__main__':
    main()