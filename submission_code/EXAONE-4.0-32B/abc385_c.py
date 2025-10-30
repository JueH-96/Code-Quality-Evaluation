import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    H = list(map(int, data[1:1+n]))
    
    groups = defaultdict(list)
    for i, h in enumerate(H):
        groups[h].append(i)
        
    ans = 1
    for h, arr in groups.items():
        m = len(arr)
        if m == 0:
            continue
        arr.sort()
        if m == 1:
            best_group = 1
        else:
            val_to_index = {}
            for idx, pos in enumerate(arr):
                val_to_index[pos] = idx
                
            dp = [[0] * m for _ in range(m)]
            best_group = 1
            for j in range(1, m):
                for i in range(0, j):
                    d = arr[j] - arr[i]
                    x = 2 * arr[i] - arr[j]
                    if x in val_to_index:
                        k = val_to_index[x]
                        if k < i:
                            dp[i][j] = dp[k][i] + 1
                        else:
                            dp[i][j] = 2
                    else:
                        dp[i][j] = 2
                    if dp[i][j] > best_group:
                        best_group = dp[i][j]
        if best_group > ans:
            ans = best_group
            
    print(ans)

if __name__ == '__main__':
    main()