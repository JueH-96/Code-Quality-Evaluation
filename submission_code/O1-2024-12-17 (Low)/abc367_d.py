def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    A = list(map(int, input().split()))

    # prefix_mod[i] will hold the prefix sum modulo M of the first i elements.
    # prefix_mod[0] = 0 (sum of zero elements).
    prefix_mod = [0] * (N + 1)
    for i in range(N):
        prefix_mod[i + 1] = (prefix_mod[i] + A[i]) % M

    # Total distance around the lake modulo M.
    tot_mod = prefix_mod[N]

    # We only consider rest areas corresponding to prefix_mod[i] for i in [0..N-1].
    # For pairs (i, j) with i < j, we need prefix_mod[i] == prefix_mod[j].
    from collections import Counter
    freq = Counter(prefix_mod[:N])

    # Count pairs (i < j) where x[i] == x[j].
    ans = 0
    for val in freq.values():
        ans += val * (val - 1) // 2

    # For pairs (j < i), we need x[j] = ( x[i] - tot_mod ) mod M.
    # We'll use a running frequency array freq2 to count how many x[j]
    # match the needed value for each i as we go.
    freq2 = [0] * M
    second_part = 0
    for i in range(N):
        needed = (prefix_mod[i] - tot_mod) % M
        second_part += freq2[needed]
        freq2[prefix_mod[i]] += 1

    ans += second_part

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()