def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    S = next(it).strip()

    # Precompute for each K-bit number if it forms a palindrome.
    # We represent a K-length block by an integer; the most significant bit is the oldest character.
    pal = [False] * (1 << K)
    for x in range(1 << K):
        is_pal = True
        for i in range(K // 2):
            if ((x >> i) & 1) != ((x >> (K - 1 - i)) & 1):
                is_pal = False
                break
        pal[x] = is_pal

    # We'll use dynamic programming over positions with state:
    # (mask, L) represents the last L characters (where L <= K-1) as bits.
    # We interpret the bits such that the most significant bit is the oldest character.
    # Initially, no characters have been chosen: state ("", length 0) represented as (0, 0).
    dp = {}
    dp[(0, 0)] = 1

    # Mapping: 'A' -> 0, 'B' -> 1.
    char_to_bit = {'A': 0, 'B': 1}

    # Process each position.
    for i in range(N):
        next_dp = {}
        ch = S[i]
        # Determine allowed choices from S[i].
        if ch == '?':
            choices = [0, 1]
        else:
            choices = [char_to_bit[ch]]
        # For each current state (mask, length) and each possible bit choice:
        for (mask, length), ways in dp.items():
            for bit in choices:
                if length < K - 1:
                    # Simply append the character.
                    new_mask = (mask << 1) | bit
                    new_len = length + 1
                    # No substring of length K has yet been formed.
                    s = (new_mask, new_len)
                    next_dp[s] = (next_dp.get(s, 0) + ways) % mod
                else:
                    # When length == K-1, appending gives a full K-length window.
                    full_window = (mask << 1) | bit  # this is a K-bit number.
                    # If the K-length block is a palindrome, skip this transition.
                    if pal[full_window]:
                        continue
                    # Otherwise, slide the window: keep the last K-1 bits.
                    new_mask = full_window & ((1 << (K - 1)) - 1)
                    new_len = K - 1
                    s = (new_mask, new_len)
                    next_dp[s] = (next_dp.get(s, 0) + ways) % mod
        dp = next_dp

    # Sum up all valid completions.
    ans = sum(dp.values()) % mod
    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()