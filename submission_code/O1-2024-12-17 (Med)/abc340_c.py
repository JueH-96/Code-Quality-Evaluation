def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    N = int(sys.stdin.readline().strip())
    
    # Memo to cache computed values of T(n)
    memo = {0: 0, 1: 0}
    
    def T(n):
        """
        Returns the total cost T(n) as described, using the identities:
        
          - T(0) = 0, T(1) = 0
          - If n is even, n = 2^k * m (m odd, k >= 1), then:
                T(n) = n + 2 * T(n // 2)
                 which can be shown to be
                T(n) = k * 2^k * m + 2^k * T(m).
          - If n is odd and n >= 3, then:
                T(n) = n + T((n - 1) // 2) + T((n + 1) // 2).
        
        We implement it with memoization to ensure we only compute each T(n) once.
        """
        if n < 2:
            return 0
        if n in memo:
            return memo[n]
        
        # Count trailing zeros to factor out 2^k from n
        # (n & -n) isolates the lowest set bit, whose bit_length()-1 is k.
        tz = (n & -n).bit_length() - 1
        
        if tz > 0:
            # n = 2^tz * m, where m is odd
            m = n >> tz  # Divide n by 2^tz
            val = tz * (1 << tz) * m + (1 << tz) * T(m)
        else:
            # n is odd (and >= 3 here)
            val = n + T(n // 2) + T(n // 2 + 1)
        
        memo[n] = val
        return val
    
    # Compute and print result
    print(T(N))

# Don't forget to call main()
if __name__ == "__main__":
    main()