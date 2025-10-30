def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))
    
    if k == 1:
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if p[i] > p[j]:
                    inversions += 1
        print(inversions % 998244353)
        return
        
    mod = 998244353
    
    def power(a, b):
        res = 1
        a %= mod
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % mod
            a = (a * a) % mod
            b //= 2
        return res
        
    def inv(n):
        return power(n, mod - 2)
        
    inv_n_minus_k_plus_1 = inv(n - k + 1)
    inv_k = inv(k)
    inv_2 = inv(2)
    
    expected_inversions = 0
    
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            u_idx = u - 1
            v_idx = v - 1
            p_u = p[u_idx]
            p_v = p[v_idx]
            
            n1 = 0
            start_i_1 = max(1, v - k + 1)
            end_i_1 = min(u, n - k + 1)
            if start_i_1 <= end_i_1:
                n1 = max(0, end_i_1 - start_i_1 + 1)
                
            n2 = max(0, min(u, v - k, n - k + 1))
            
            n3 = 0
            start_i_3 = max(u + 1, 1)
            end_i_3 = min(v, n - k + 1)
            if start_i_3 <= end_i_3:
                n3 = max(0, end_i_3 - start_i_3 + 1)
                
            n4a = max(0, n - k - v + 1) if v <= n - k else 0
            n4b = max(0, u - k) if u >= k + 1 else 0
            n4 = n4a + n4b
            
            sum1 = 0
            start_i_2 = 1
            end_i_2 = min(u, v - k, n - k + 1)
            for i in range(start_i_2, end_i_2 + 1):
                count = 0
                for j in range(i - 1, i + k - 1):
                    if p[j] > p_v:
                        count += 1
                sum1 += count
                
            sum2 = 0
            start_i_3_calc = max(u + 1, 1)
            end_i_3_calc = min(v, n - k + 1)
            for i in range(start_i_3_calc, end_i_3_calc + 1):
                count = 0
                for j in range(i - 1, i + k - 1):
                    if p[j] < p_u:
                        count += 1
                sum2 += count
                
            i_uv = 1 if p_u > p_v else 0
            
            prob_uv = (n1 * inv_2 + sum1 * inv_k + sum2 * inv_k + i_uv * n4) % mod
            prob_uv = (prob_uv * inv_n_minus_k_plus_1) % mod
            expected_inversions = (expected_inversions + prob_uv) % mod
            
    print(expected_inversions)

if __name__ == '__main__':
    solve()