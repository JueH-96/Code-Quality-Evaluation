class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        mod = 10**9 + 7
        # We want arrays of length n (each element in [1, m])
        # that have exactly k indices i (1 <= i < n) with arr[i-1] == arr[i].
        # Observe that in any such array the number of adjacent equal pairs is:
        #    count = (length of array) - (number of segments)
        # That is, if the array is partitioned into segments of consecutive equal elements,
        # then each segment of length L contributes L - 1 equal-adjacencies.
        # Total equal-adjacencies = n - (number of segments) = k.
        # Hence, number of segments = n - k.
        #
        # Now, how can we build such an array?
        # 1. Choose the positions where a neighboring change happens. Out of the n-1 
        #    gaps between consecutive elements, exactly (number of segments - 1) 
        #    must be changes. That is, choose (n-k-1) positions among (n-1).
        #    The number of ways to choose this is C(n-1, n-k-1). (Note that C(n-1, n-k-1) is
        #    equivalent to C(n-1, k).)
        #
        # 2. Fill in the segments:
        #    - The first segment can be filled with any of the m values.
        #    - Every subsequent segment must start with a value that is different from the previous one.
        #      Thus, for each such segment there are (m-1) choices.
        #
        # Therefore, the answer is:
        #   answer = C(n-1, n-k-1) * m * (m-1)^(n-k-1)
        # We return the answer modulo mod.
        
        # Precompute factorials and inverse factorials modulo mod up to (n-1)
        max_val = n - 1
        fact = [1] * (max_val + 1)
        for i in range(2, max_val + 1):
            fact[i] = fact[i - 1] * i % mod
        
        invfact = [1] * (max_val + 1)
        invfact[max_val] = pow(fact[max_val], mod - 2, mod)
        for i in range(max_val, 0, -1):
            invfact[i - 1] = invfact[i] * i % mod
        
        # Calculate binomial coefficient C(n-1, n-k-1) which equals C(n-1, k)
        if k > max_val:
            return 0  # though k is guaranteed to be <= n-1
        choose = (fact[max_val] * invfact[k] % mod) * invfact[max_val - k] % mod
        
        exponent = n - k - 1  # exponent for (m - 1)
        return choose * m % mod * pow(m - 1, exponent, mod) % mod

# The following solve() function uses the above implementation to work with input/output.
def solve():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n, m, k = map(int, data[:3])
    sol = Solution()
    result = sol.countGoodArrays(n, m, k)
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    solve()
    
# Example test runs:
if __name__ == '__main__' and False:
    sol = Solution()
    print(sol.countGoodArrays(3, 2, 1))  # Expected output: 4
    print(sol.countGoodArrays(4, 2, 2))  # Expected output: 6
    print(sol.countGoodArrays(5, 2, 0))  # Expected output: 2