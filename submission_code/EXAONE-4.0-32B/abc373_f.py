import collections
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    W_val = int(data[1])
    items = []
    index = 2
    for i in range(n):
        w = int(data[index]); v = int(data[index+1]); index += 2
        items.append((w, v))
    
    NEG_INF = -10**18
    dp = [NEG_INF] * (W_val+1)
    dp[0] = 0
    
    for wi, vi in items:
        next_dp = [NEG_INF] * (W_val+1)
        for r in range(wi):
            indices = []
            j = r
            while j <= W_val:
                indices.append(j)
                j += wi
            g = [dp[j] for j in indices]
            T = len(g)
            if T == 0:
                continue
                
            Q = collections.deque()
            new_g = [NEG_INF] * T
            
            for t in range(T):
                m0 = 2 * t
                c0 = g[t] - t * vi - t*t
                
                while len(Q) >= 2:
                    L1 = Q[-1]
                    L2 = Q[-2]
                    A = (c0 - L2[1]) * (L1[0] - L2[0])
                    B = (L1[1] - L2[1]) * (m0 - L2[0])
                    if A >= B:
                        Q.pop()
                    else:
                        break
                Q.append((m0, c0))
                
                while len(Q) >= 2:
                    val0 = Q[0][0] * t + Q[0][1]
                    val1 = Q[1][0] * t + Q[1][1]
                    if val0 <= val1:
                        Q.popleft()
                    else:
                        break
                        
                best_val = Q[0][0] * t + Q[0][1]
                new_g[t] = best_val + t * vi - t*t
                
            for idx, j in enumerate(indices):
                next_dp[j] = new_g[idx]
                
        dp = next_dp
        
    ans = max(dp)
    print(ans)

if __name__ == '__main__':
    main()