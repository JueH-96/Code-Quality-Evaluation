def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    S = data[0].strip()
    N = int(data[1])
    L = len(S)

    # Precompute powers of two: pow2[k] = 2^k
    pow2 = [1 << i for i in range(L+1)]

    # minimal bit at each position j: if S[j]=='1' then 1, else 0
    minimal = [1 if c == '1' else 0 for c in S]

    # suf_min[j] = sum of minimal[k]*2^(L-1-k) for k = j..L-1
    suf_min = [0] * (L + 1)
    for j in range(L - 1, -1, -1):
        weight = pow2[L - 1 - j]
        suf_min[j] = minimal[j] * weight + suf_min[j + 1]

    minimal_t = suf_min[0]

    # max_t: replace every '?' with '1'
    max_t = 0
    for j, c in enumerate(S):
        if c == '1' or c == '?':
            max_t += pow2[L - 1 - j]

    # If the maximal possible T-value is <= N, answer is max_t
    if max_t <= N:
        print(max_t)
        return

    # If the minimal possible T-value is > N, no solution
    if minimal_t > N:
        print(-1)
        return

    # Otherwise do a greedy from MSB to LSB
    prefix_value = 0
    for i, c in enumerate(S):
        weight = pow2[L - 1 - i]
        if c == '?':
            # try setting this bit to 1
            cand = prefix_value + weight
            # check if minimal completion of suffix keeps <= N
            if cand + suf_min[i + 1] <= N:
                prefix_value = cand
            # else leave it 0
        elif c == '1':
            prefix_value += weight
        # if c == '0', we add nothing

    print(prefix_value)

if __name__ == "__main__":
    main()