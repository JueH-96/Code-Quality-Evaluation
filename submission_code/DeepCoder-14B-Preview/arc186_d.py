from functools import lru_cache

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    A = [0] + A  # Convert to 1-based index

    @lru_cache(maxsize=None)
    def dp(pos, groups_left, tight):
        if pos > N:
            return 0
        if pos == N:
            return 1 if groups_left == 0 else 0
        res = 0
        if groups_left == 0:
            max_v = N - pos
            for v in range(0, max_v + 1):
                if v == 0 and pos != N:
                    continue
                if tight:
                    if v > A[pos]:
                        continue
                    new_tight = (v == A[pos])
                else:
                    new_tight = 0
                res += dp(pos + 1, v, new_tight)
        else:
            max_val = N - pos
            for val in range(0, max_val + 1):
                if tight:
                    if val > A[pos]:
                        continue
                    new_tight = (val == A[pos])
                else:
                    new_tight = 0
                res += dp(pos + 1, groups_left, new_tight)
        return res % MOD

    print(dp(1, 0, 1) % MOD)

if __name__ == '__main__':
    main()