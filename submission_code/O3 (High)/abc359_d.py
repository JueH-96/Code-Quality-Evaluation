import sys

MOD = 998244353

def main() -> None:
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # pre-compute palindromes of length K
    max_window = 1 << K
    is_pal = [False] * max_window
    for x in range(max_window):
        ok = True
        for i in range(K // 2):
            if ((x >> i) & 1) != ((x >> (K - 1 - i)) & 1):
                ok = False
                break
        is_pal[x] = ok

    MASK_ALL = (1 << (K - 1)) - 1          # keeps K-1 least significant bits

    dp_cur = [0] * (1 << (K - 1))
    dp_cur[0] = 1                           # empty prefix

    for pos in range(N):
        dp_nxt = [0] * (1 << (K - 1))
        chars = []
        if S[pos] == '?':
            chars = [0, 1]                  # A and B
        else:
            chars = [0] if S[pos] == 'A' else [1]

        for mask, ways in enumerate(dp_cur):
            if ways == 0:
                continue
            for b in chars:
                window_bits = (mask << 1) | b          # K bits (some leading zeros)
                if pos >= K - 1 and is_pal[window_bits]:
                    continue                           # bad window
                new_mask = window_bits & MASK_ALL
                dp_nxt[new_mask] = (dp_nxt[new_mask] + ways) % MOD
        dp_cur = dp_nxt

    print(sum(dp_cur) % MOD)


if __name__ == "__main__":
    main()