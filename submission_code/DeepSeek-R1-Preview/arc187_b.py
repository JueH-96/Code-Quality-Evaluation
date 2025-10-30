MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:2+N]))
    
    # Precompute q_pre: number of -1s before each index
    q_pre = [0] * N
    count = 0
    for i in range(N):
        q_pre[i] = count
        if B[i] == -1:
            count += 1
    
    # Precompute q_post: number of -1s after each index
    q_post = [0] * N
    count = 0
    for i in range(N-1, -1, -1):
        q_post[i] = count
        if B[i] == -1:
            count += 1
    
    # Precompute M^q_post[i] mod MOD for all i
    M_pow_q_post = [pow(M, q_post[i], MOD) for i in range(N)]
    
    total = 0
    for i in range(N):
        # Collect fixed elements before i
        fixed_lt_i = [B[j] for j in range(i) if B[j] != -1]
        if B[i] != -1:
            x = B[i]
            # Check if x is less than all fixed elements before i
            valid = True
            for b in fixed_lt_i:
                if b <= x:
                    valid = False
                    break
            if not valid:
                continue
            # Compute count_pre = (M - x) ** q_pre[i]
            count_pre = pow(M - x, q_pre[i], MOD)
            # Multiply by M^q_post[i]
            contrib = (count_pre * M_pow_q_post[i]) % MOD
            total = (total + contrib) % MOD
        else:
            # B[i] is -1, x can vary from 1 to M
            if not fixed_lt_i:
                # All x from 1 to M are valid
                x_min = 1
                x_max = M
            else:
                x_min = 1
                x_max = min(fixed_lt_i) - 1
                if x_max < x_min:
                    continue  # no valid x
            # Sum (M - x) ** q_pre[i] for x in [x_min, x_max]
            sum_pre = 0
            for x in range(x_min, x_max + 1):
                term = pow(M - x, q_pre[i], MOD)
                sum_pre = (sum_pre + term) % MOD
            # Multiply by M^q_post[i]
            contrib = (sum_pre * M_pow_q_post[i]) % MOD
            total = (total + contrib) % MOD
    
    print(total % MOD)

if __name__ == "__main__":
    main()