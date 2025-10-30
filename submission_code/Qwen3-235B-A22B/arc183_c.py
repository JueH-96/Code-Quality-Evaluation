import sys
input = sys.stdin.read

def main():
    MOD = 998244353
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    C = [[] for _ in range(N+2)]  # C[x] contains (L, R) for constraints with X=x
    
    for _ in range(M):
        L = int(data[idx])
        R = int(data[idx+1])
        X = int(data[idx+2])
        idx += 3
        C[X].append((L, R))
    
    occupied = [False] * (N + 2)  # 1-based indexing
    ans = 1
    
    for K in range(N, 0, -1):  # from N downto 1
        # Build prefix sum_occ
        sum_occ = [0] * (N + 2)
        for i in range(1, N+1):
            sum_occ[i] = sum_occ[i-1] + (1 if occupied[i] else 0)
        
        valid_positions = 0
        # Check each position p
        for p in range(1, N+1):
            if occupied[p]:
                continue
            valid = True
            # Check all constraints for this p
            for (L, R) in C[p]:
                if (sum_occ[R] - sum_occ[L-1]) == 0:
                    valid = False
                    break
            if valid:
                valid_positions += 1
        
        if valid_positions == 0:
            print(0)
            return
        
        ans = (ans * valid_positions) % MOD
        
        # Find the first valid p to occupy
        chosen = -1
        for p in range(1, N+1):
            if not occupied[p]:
                valid = True
                for (L, R) in C[p]:
                    if (sum_occ[R] - sum_occ[L-1]) == 0:
                        valid = False
                        break
                if valid:
                    chosen = p
                    break
        occupied[chosen] = True
    
    print(ans % MOD)

if __name__ == "__main__":
    main()