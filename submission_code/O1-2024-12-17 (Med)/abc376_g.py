def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    MOD = 998244353
    
    # Fast modular inverse using Fermat's little theorem (MOD is prime)
    def modinv(x):
        return pow(x, MOD-2, MOD)
    
    t_index = 0
    T = int(input_data[t_index]); t_index += 1
    
    answers = []
    
    for _ in range(T):
        N = int(input_data[t_index]); t_index += 1
        
        # Read parents (p_i)
        p = [0]*N
        for i in range(N):
            p[i] = int(input_data[t_index])
            t_index += 1
        
        # Read a_i (1-based for convenience)
        a = [0]*(N+1)
        for i in range(1, N+1):
            a[i] = int(input_data[t_index])
            t_index += 1
        
        # Build tree adjacency and incoming counts
        children = [[] for _ in range(N+1)]
        incoming = [0]*(N+1)
        
        # p[i-1] is the parent of vertex i
        # If p[i-1] == 0, that vertex i is initially "available"
        for i in range(1, N+1):
            pa = p[i-1]
            if pa == 0:
                incoming[i] = 0   # root's child => available
                # We'll just store them under children[0] for convenience
                children[0].append(i)
            else:
                incoming[i] = 1   # must wait until pa is searched
                children[pa].append(i)
        
        # Total of all a[i]
        total_a = sum(a[1:])  # a[1]..a[N]
        total_mod = total_a % MOD
        
        # Priority queue (max-heap) of (a_i, i). Python heapq is min-heap => use negative
        heap = []
        for c0 in children[0]:
            heapq.heappush(heap, (-a[c0], c0))
        
        pos = 0
        sum_a_pos = 0  # will hold Σ a_i * position(i), all modded
        
        # Greedy topological order picking the largest a_i among available
        while heap:
            neg_val, node = heapq.heappop(heap)
            val = -neg_val
            pos += 1
            # Accumulate contribution
            sum_a_pos = (sum_a_pos + val * pos) % MOD
            
            # Unlock children
            for nxt in children[node]:
                incoming[nxt] -= 1
                if incoming[nxt] == 0:
                    heapq.heappush(heap, (-a[nxt], nxt))
        
        # Expected cost = (sum_a_pos / total_a) mod 998244353
        # = sum_a_pos * inv(total_a) mod 998244353
        inv_total = modinv(total_mod)
        ans = (sum_a_pos * inv_total) % MOD
        answers.append(ans)
    
    print('
'.join(map(str, answers)))