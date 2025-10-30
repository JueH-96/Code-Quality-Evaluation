import math

class Solution:
    def waysToReachStair(self, k: int) -> int:
        # We count sequences of n up‐moves (U) and m down‐moves (D)
        # with no two consecutive D's, where
        #   total climb = (sum of 2^i for i=0..n-1) = 2^n - 1
        #   net position = 1 + (2^n - 1) - m = 2^n - m
        # We require 2^n - m = k  =>  m = 2^n - k,  m >= 0
        # And no two D's consecutive among n U's gives C(n+1, m) ways
        ways = 0
        n = 0
        while True:
            t = 1 << n       # 2^n
            m = t - k
            # If m > n+1, then t - k > n+1  <=>  t > k + n+1
            # For larger n, t grows, so we can break
            if m > n + 1:
                break
            if m >= 0:
                # count ways to choose m down‐moves in n+1 gaps
                ways += math.comb(n+1, m)
            n += 1
        return ways

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.waysToReachStair(0))  # 2
    print(sol.waysToReachStair(1))  # 4