# YOUR CODE HERE
from typing import List
from functools import lru_cache

def solve(n: int, m: int, conditions: List[List[int]]) -> int:
    MOD = 998244353

    @lru_cache(None)
    def dp(index: int, forbidden: int) -> int:
        if index == n + 1:
            return 1
        count = 0
        for i in range(1, n + 1):
            if i == forbidden:
                continue
            valid = True
            for L, R, X in conditions:
                if L <= index <= R and i == X:
                    valid = False
                    break
            if valid:
                count += dp(index + 1, i)
                count %= MOD
        return count

    return dp(1, 0)

if __name__ == "__main__":
    n, m = map(int, input().split())
    conditions = [list(map(int, input().split())) for _ in range(m)]
    print(solve(n, m, conditions))