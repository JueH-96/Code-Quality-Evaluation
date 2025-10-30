def main():
    import sys

    # Read input
    input = sys.stdin.read
    W_B = input().strip().split()
    W = int(W_B[0])
    B = int(W_B[1])

    # Define the pattern
    period = "wbwbwwbwbwbw"
    len_period = len(period)

    # Compute prefix sums for 'w' and 'b'
    prefix_w = [0] * (len_period + 1)
    prefix_b = [0] * (len_period + 1)
    for i in range(1, len_period + 1):
        prefix_w[i] = prefix_w[i-1] + (1 if period[i-1] == 'w' else 0)
        prefix_b[i] = prefix_b[i-1] + (1 if period[i-1] == 'b' else 0)

    # Precompute all possible suffixes (from l to 12)
    suffixes = []
    for l in range(1, len_period + 1):
        w_suf = prefix_w[len_period] - prefix_w[l-1]
        b_suf = prefix_b[len_period] - prefix_b[l-1]
        suffixes.append((w_suf, b_suf))

    # Precompute all possible prefixes (from 1 to r)
    prefixes = []
    for r in range(1, len_period + 1):
        w_pre = prefix_w[r]
        b_pre = prefix_b[r]
        prefixes.append((w_pre, b_pre))

    # Check for the existence of the substring
    found = False
    for suf in suffixes:
        for pre in prefixes:
            if (W - suf[0] - pre[0]) % 7 == 0:
                K = (W - suf[0] - pre[0]) // 7
                if K >= 0 and suf[1] + K * 5 + pre[1] == B:
                    found = True
                    print("Yes")
                    return
    if not found:
        print("No")

if __name__ == "__main__":
    main()