def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    # Compute prefix sums mod M
    # c[i] = (A[0] + A[1] + ... + A[i-1]) mod M
    # c[0] = 0, c[N] = sum(A) mod M
    c = [0] * (N + 1)
    for i in range(N):
        c[i + 1] = (c[i] + A[i]) % M

    # Case 1: t > s => distance(s, t) = prefix[t] - prefix[s], 
    # we want (c[t] - c[s]) mod M = 0 => c[t] = c[s].
    # We'll count how many pairs (s, t) with s < t have c[s] = c[t].
    freq = [0] * M
    for i in range(N):
        freq[c[i]] += 1

    ans_case1 = 0
    for k in freq:
        if k > 1:
            ans_case1 += k * (k - 1) // 2

    # Case 2: t < s => distance(s, t) 
    # = prefix[N] - (prefix[s] - prefix[t]) = c[N] + (c[t] - c[s]) (mod M).
    # We want c[N] + (c[t] - c[s]) = 0 (mod M) => c[t] - c[s] = -c[N] (mod M).
    # Define d = (M - c[N]) mod M so that c[t] = c[s] + d (mod M).
    # We'll count for each s, how many t < s satisfy c[t] = (c[s] + d) mod M.
    d = (M - c[N]) % M
    ans_case2 = 0
    freq_t = [0] * M  # freq_t[x] counts how many indices t < s with c[t] = x

    for s in range(N):
        needed_val = (c[s] + d) % M
        ans_case2 += freq_t[needed_val]
        freq_t[c[s]] += 1

    print(ans_case1 + ans_case2)

def main():
    solve()

if __name__ == "__main__":
    main()