class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        graph = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            graph[p].append(i)
        
        for i in range(n):
            graph[i].sort()
        
        order_arr = [''] * n
        seg_start = [-1] * n
        seg_end = [-1] * n
        time = 0
        stack = [(0, 0)]
        
        while stack:
            node, idx = stack.pop()
            if idx == 0:
                seg_start[node] = time
            if idx < len(graph[node]):
                next_child = graph[node][idx]
                stack.append((node, idx + 1))
                stack.append((next_child, 0))
            else:
                order_arr[time] = s[node]
                seg_end[node] = time
                time += 1
        
        rev_order_arr = order_arr[::-1]
        
        mod1 = 10**9 + 7
        mod2 = 10**9 + 9
        base_val = 131
        
        def precompute(arr, mod_val, base_val):
            n_arr = len(arr)
            H = [0] * (n_arr + 1)
            pow_base = [1] * (n_arr + 1)
            for i in range(1, n_arr + 1):
                pow_base[i] = (pow_base[i - 1] * base_val) % mod_val
            for i in range(n_arr):
                val = ord(arr[i]) - ord('a') + 1
                H[i + 1] = (H[i] * base_val + val) % mod_val
            return H, pow_base
        
        H1, pow1_1 = precompute(order_arr, mod1, base_val)
        H2, pow1_2 = precompute(order_arr, mod2, base_val)
        H1_rev, pow2_1 = precompute(rev_order_arr, mod1, base_val)
        H2_rev, pow2_2 = precompute(rev_order_arr, mod2, base_val)
        
        def get_hash(H, pow_base, l, r, mod_val):
            len_seg = r - l + 1
            h = (H[r + 1] - (H[l] * pow_base[len_seg]) % mod_val) % mod_val
            if h < 0:
                h += mod_val
            return h
        
        ans = [False] * n
        for i in range(n):
            l = seg_start[i]
            r = seg_end[i]
            len_seg = r - l + 1
            l_rev = n - 1 - r
            r_rev = n - 1 - l
            
            h1_forward = get_hash(H1, pow1_1, l, r, mod1)
            h2_forward = get_hash(H2, pow1_2, l, r, mod2)
            h1_rev = get_hash(H1_rev, pow2_1, l_rev, r_rev, mod1)
            h2_rev = get_hash(H2_rev, pow2_2, l_rev, r_rev, mod2)
            
            if h1_forward == h1_rev and h2_forward == h2_rev:
                ans[i] = True
        
        return ans