def main():
    import sys
    import heapq

    MOD = 998244353

    # Fast modular inverse (Fermat's Little Theorem since MOD is prime)
    def inv(x):
        return pow(x, MOD-2, MOD)

    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1

    # We'll accumulate answers to print at once
    answers = []

    for _ in range(t):
        N = int(input_data[idx]); idx += 1

        # Read parent array (p_i), 0-based in input_data but child is i+1
        parents = list(map(int, input_data[idx:idx+N]))
        idx += N

        # Read a_i array
        arr = list(map(int, input_data[idx:idx+N]))
        idx += N

        # Build adjacency (children) list for the tree
        children = [[] for _ in range(N+1)]
        for i in range(N):
            p = parents[i]
            child = i + 1
            children[p].append(child)

        # a array with an offset: a[0] = 0, a[i] = arr[i-1] for i=1..N
        a = [0] * (N+1)
        for i in range(1, N+1):
            a[i] = arr[i-1]

        # Compute subtree sums P[v] by iterative postorder from root=0
        visited = [False]*(N+1)
        stack = [0]
        postorder = []

        while stack:
            v = stack[-1]
            if not visited[v]:
                visited[v] = True
                # Push children if not visited
                for c in children[v]:
                    if not visited[c]:
                        stack.append(c)
            else:
                stack.pop()
                postorder.append(v)

        # Now P[v] = sum of a[x] over the subtree of v
        P = [0]*(N+1)
        for v in postorder:
            subtotal = a[v]
            for c in children[v]:
                subtotal += P[c]
            P[v] = subtotal

        # Perform a "best-first" search from the children of 0, using max-heap by P[v]
        # The position of each node v in this order is the number of expansions to discover v.
        heap = []
        for c in children[0]:
            heapq.heappush(heap, (-P[c], c))

        position = [0]*(N+1)
        counter = 0
        while heap:
            neg_val, node = heapq.heappop(heap)
            counter += 1
            position[node] = counter
            for c in children[node]:
                heapq.heappush(heap, (-P[c], c))

        # Expected expansions = (1 / sum(a_i)) * Σ (a[node] * position[node])
        # We'll do it all mod 998244353
        total_sum = 0
        for v in range(1, N+1):
            total_sum = (total_sum + a[v] * position[v]) % MOD

        # P[0] = sum_{v=1..N} a[v]
        big_sum = P[0] % MOD
        ans = (total_sum * inv(big_sum)) % MOD
        answers.append(str(ans))

    print("
".join(answers))

# Call main() at the end:
# (Do not remove this call, it is required)
main()