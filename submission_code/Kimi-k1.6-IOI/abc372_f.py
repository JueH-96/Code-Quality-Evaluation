MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    K = int(data[idx])
    idx += 1
    
    adj_extra = {}
    for _ in range(M):
        X = int(data[idx])
        idx += 1
        Y = int(data[idx])
        idx += 1
        if X not in adj_extra:
            adj_extra[X] = []
        adj_extra[X].append(Y)
    
    prev_dp = {1: 1}
    
    for _ in range(K):
        curr_dp = {}
        for u in prev_dp:
            count = prev_dp[u]
            # Cycle edge transition
            v_cycle = (u % N) + 1
            curr_dp[v_cycle] = curr_dp.get(v_cycle, 0) + count
            # Extra edges transitions
            if u in adj_extra:
                for v in adj_extra[u]:
                    curr_dp[v] = curr_dp.get(v, 0) + count
        # Apply modulo and filter out zeros
        new_curr_dp = {}
        for v, cnt in curr_dp.items():
            cnt_mod = cnt % MOD
            if cnt_mod != 0:
                new_curr_dp[v] = cnt_mod
        prev_dp = new_curr_dp
    
    total = sum(prev_dp.values()) % MOD
    print(total)

if __name__ == "__main__":
    main()