def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, W = map(int, input_data[:2])
    wv = input_data[2:]
    w_list = []
    v_list = []
    ptr = 0
    for _ in range(N):
        w_list.append(int(wv[ptr]))
        v_list.append(int(wv[ptr+1]))
        ptr += 2

    # dp[x] will hold the maximum total happiness achievable with capacity x
    dp = [0]*(W+1)

    for i in range(N):
        w_i = w_list[i]
        v_i = v_list[i]

        # Maximum number of items k_i we might possibly take from this type
        # is limited by the bag capacity, but also by where incremental gain goes non-positive.
        # The happiness from picking k_i items is k_i*v_i - k_i^2, which is maximized roughly
        # around k_i = v_i/2; once v_i - (2k - 1) becomes negative, adding more items reduces total happiness.
        max_k = min(W // w_i, (v_i + 1) // 2)

        if max_k <= 0:
            # No point in adding blocks if we can't take even 1 item profitably
            continue

        # Precompute SSR array: SSR[k] = sum_{r=1..k} (v_i - (2r-1)) = k*v_i - k^2
        # SSR[0] = 0
        SSR = [0]*(max_k+1)
        for k in range(1, max_k+1):
            SSR[k] = k*v_i - k*k

        # We will decompose max_k into sums of powers of 2: 1,2,4,... plus remainder
        # Then treat each block as a "0-1 knap-sack item" with weight=block_size*w_i
        # and value = SSR[L+block_size] - SSR[L], where L is the start of that block.
        # This is the standard binary decomposition trick for multiple knapsack.
        current_k = max_k
        start_index = 0
        block_size = 1
        blocks = []

        while block_size <= current_k:
            weight_block = block_size * w_i
            value_block = SSR[start_index+block_size] - SSR[start_index]
            blocks.append((weight_block, value_block))
            start_index += block_size
            current_k -= block_size
            block_size <<= 1

        if current_k > 0:  # remainder
            weight_block = current_k * w_i
            value_block = SSR[start_index+current_k] - SSR[start_index]
            blocks.append((weight_block, value_block))

        # Now do 0-1 knapsack on these blocks in descending order of capacity
        for (bw, bv) in blocks:
            for cap in range(W, bw-1, -1):
                new_val = dp[cap - bw] + bv
                if new_val > dp[cap]:
                    dp[cap] = new_val

    print(dp[W])

def main():
    solve()

if __name__ == "__main__":
    main()