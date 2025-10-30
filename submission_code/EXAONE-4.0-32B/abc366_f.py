import sys

def main():
    data = sys.stdin.read().split()
    if not data:
         return
    
    n = int(data[0])
    K_val = int(data[1])
    functions = []
    idx = 2
    for i in range(n):
        a = int(data[idx])
        b = int(data[idx+1])
        idx += 2
        functions.append((a, b))
        
    if K_val == 0:
        print(1)
        return
        
    from collections import defaultdict
    group_count = defaultdict(int)
    for a, b in functions:
        group_count[(a, b)] += 1
        
    groups = []
    for (a, b), cnt in group_count.items():
        r = (1.0 - a) / b
        groups.append((a, b, cnt, r))
        
    groups.sort(key=lambda x: x[3], reverse=True)
    
    dp = [-10**18] * (K_val + 1)
    dp[0] = 1
    
    for (a, b, cnt, r) in groups:
        max_m = min(cnt, K_val)
        fac_list = []
        con_list = []
        pow_a = 1
        for m in range(1, max_m + 1):
            pow_a = pow_a * a
            if a == 1:
                const_val = b * m
            else:
                const_val = b * (pow_a - 1) // (a - 1)
            fac_list.append(pow_a)
            con_list.append(const_val)
            
        new_dp = dp[:]
        for k in range(0, K_val + 1):
            if dp[k] == -10**18:
                continue
            for idx_m in range(len(fac_list)):
                m = idx_m + 1
                if k + m > K_val:
                    break
                fac = fac_list[idx_m]
                con = con_list[idx_m]
                candidate = fac * dp[k] + con
                if candidate > new_dp[k + m]:
                    new_dp[k + m] = candidate
        dp = new_dp
        
    print(dp[K_val])

if __name__ == '__main__':
    main()