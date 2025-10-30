import sys
import heapq

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        p_list = list(map(int, input[ptr:ptr+N]))
        ptr += N
        a_list = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        # Build children list
        children = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            parent = p_list[i-1]
            children[parent].append(i)
        
        # Compute S[i]
        S = [0] * (N + 1)
        for i in range(1, N+1):
            S[i] = a_list[i-1]
        
        # Process nodes in reverse order to compute S
        for i in range(N, 0, -1):
            for c in children[i]:
                S[i] += S[c]
        
        # Build heap and compute t[i]
        heap = []
        for child in children[0]:
            heapq.heappush(heap, (-S[child], child))
        
        current_time = 1
        t = [0] * (N + 1)
        while heap:
            neg_s, u = heapq.heappop(heap)
            t[u] = current_time
            current_time += 1
            for v in children[u]:
                heapq.heappush(heap, (-S[v], v))
        
        # Calculate sum_a_t_mod and total_A_mod
        sum_a_t_mod = 0
        total_A_mod = 0
        for i in range(N):
            ai = a_list[i]
            ti = t[i+1]
            sum_a_t_mod = (sum_a_t_mod + ai * ti) % MOD
            total_A_mod = (total_A_mod + ai) % MOD
        
        # Compute modular inverse and final answer
        inv_total_A = pow(total_A_mod, MOD-2, MOD)
        ans = (sum_a_t_mod * inv_total_A) % MOD
        print(ans)

if __name__ == "__main__":
    main()