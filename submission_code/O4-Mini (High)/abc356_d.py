def main():
    import sys
    data = sys.stdin.readline().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    mod = 998244353
    T = N + 1
    ans = 0
    # For each bit position i, if M has a 1 there, we add the count of 1s
    # at bit i among all k from 0 to N.
    for i in range(60):
        if ((M >> i) & 1) == 0:
            continue
        # period = 2^(i+1)
        # full_cycles = T // period
        # each full cycle contributes 2^i ones at bit i
        full_cycles = T >> (i + 1)
        ones_full = full_cycles << i
        # remainder of numbers after full cycles
        rem = T & ((1 << (i + 1)) - 1)
        # in the partial cycle, the first 2^i are zeros, then up to rem-2^i ones
        extra = rem - (1 << i)
        if extra < 0:
            extra = 0
        total_i = ones_full + extra
        ans = (ans + total_i) % mod
    print(ans)

# call main to execute
main()