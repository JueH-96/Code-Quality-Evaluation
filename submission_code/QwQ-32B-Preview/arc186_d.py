from functools import lru_cache

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    MOD = 998244353

    @lru_cache(maxsize=None)
    def dp(pos, required, is_less):
        if pos == N:
            return 1 if required == 0 else 0
        if required == 0:
            return 0
        res = 0
        upper = A[pos] if not is_less else A[pos] - 1
        for v in range(0, upper + 1):
            if v > N - pos - 1:
                continue
            new_required = required + v - 1
            if new_required < 0:
                continue
            res = (res + dp(pos + 1, new_required, is_less or (v < A[pos]))) % MOD
        return res

    print(dp(0, 1, False))

if __name__ == '__main__':
    main()