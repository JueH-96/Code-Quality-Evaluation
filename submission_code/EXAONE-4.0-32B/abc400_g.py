import sys
from functools import lru_cache

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        k = int(data[index+1])
        index += 2
        cakes = []
        for i in range(n):
            x = int(data[index])
            y = int(data[index+1])
            z = int(data[index+2])
            index += 3
            cakes.append((x, y, z))
        
        if n <= 10:
            @lru_cache(maxsize=None)
            def dfs(mask):
                cnt_ones = bin(mask).count('1')
                count_pairs = cnt_ones // 2
                if count_pairs == k:
                    return 0
                remaining_cakes = n - cnt_ones
                if remaining_cakes < 2 * (k - count_pairs):
                    return -10**18
                i = 0
                while i < n:
                    if not (mask & (1 << i)):
                        break
                    i += 1
                if i >= n:
                    return -10**18
                res = dfs(mask | (1 << i))
                for j in range(i+1, n):
                    if mask & (1 << j):
                        continue
                    x_i, y_i, z_i = cakes[i]
                    x_j, y_j, z_j = cakes[j]
                    val = max(x_i + x_j, y_i + y_j, z_i + z_j)
                    new_mask = mask | (1 << i) | (1 << j)
                    candidate = val + dfs(new_mask)
                    if candidate > res:
                        res = candidate
                return res
            
            ans = dfs(0)
            results.append(str(ans))
            dfs.cache_clear()
        else:
            ans_candidate = -10**18
            for mask in range(8):
                arr = []
                for i in range(n):
                    x, y, z = cakes[i]
                    total_val = 0
                    if mask & 1:
                        total_val += x
                    else:
                        total_val -= x
                    if mask & 2:
                        total_val += y
                    else:
                        total_val -= y
                    if mask & 4:
                        total_val += z
                    else:
                        total_val -= z
                    arr.append(total_val)
                arr.sort(reverse=True)
                total_sum = sum(arr[:2*k])
                if total_sum > ans_candidate:
                    ans_candidate = total_sum
            results.append(str(ans_candidate))
    
    print("
".join(results))

if __name__ == "__main__":
    main()