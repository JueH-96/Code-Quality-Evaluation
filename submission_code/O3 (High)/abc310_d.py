import sys
sys.setrecursionlimit(10000)

def main() -> None:
    N, T, M = map(int, sys.stdin.readline().split())
    bad = [0] * N                  # bitmask of players incompatible with i
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        a -= 1
        b -= 1
        bad[a] |= 1 << b
        bad[b] |= 1 << a

    # pre-compute whether a subset is a valid team
    good = [True] * (1 << N)
    good[0] = False
    for mask in range(1, 1 << N):
        ok = True
        m = mask
        while m:
            i = (m & -m).bit_length() - 1
            if bad[i] & mask:          # i meets an incompatible partner inside mask
                ok = False
                break
            m &= m - 1
        good[mask] = ok

    from functools import lru_cache

    @lru_cache(None)
    def dfs(rem_mask: int, k: int) -> int:
        """number of ways to split `rem_mask` into `k` non-empty good groups"""
        if k == 1:
            return 1 if good[rem_mask] else 0
        if rem_mask == 0 or (rem_mask.bit_count() < k):
            return 0

        # pick smallest unassigned player
        first = (rem_mask & -rem_mask).bit_length() - 1
        rest = rem_mask ^ (1 << first)

        ans = 0
        sub = rest
        while True:                                # iterate through all subsets of 'rest'
            team = sub | (1 << first)              # current group
            if good[team]:
                rem = rem_mask ^ team
                if rem.bit_count() >= k - 1:       # enough players left
                    ans += dfs(rem, k - 1)
            if sub == 0:
                break
            sub = (sub - 1) & rest                 # next subset
        return ans

    print(dfs((1 << N) - 1, T))


if __name__ == "__main__":
    main()