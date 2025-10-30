class Solution:
    def minAnagramLength(self, s: str) -> int:
        # Length of the full string
        n = len(s)
        # Precompute total character counts in s
        total_counts = [0] * 26
        for ch in s:
            total_counts[ord(ch) - ord('a')] += 1
        
        # Generate all divisors of n (possible chunk lengths), sorted ascending
        divisors = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.append(i)
                if i * i != n:
                    divisors.append(n // i)
            i += 1
        divisors.sort()
        
        # Helper to test if chunk length L works
        def can_use_length(L: int) -> bool:
            k = n // L  # number of chunks
            # Quick check: each total letter count must be divisible by k
            for cnt in total_counts:
                if cnt % k != 0:
                    return False
            # Compute the reference frequency from the first chunk
            ref = [0] * 26
            for idx in range(L):
                ref[ord(s[idx]) - ord('a')] += 1
            # Check all other chunks
            for start in range(L, n, L):
                cnt = [0] * 26
                end = start + L
                for idx in range(start, end):
                    cnt[ord(s[idx]) - ord('a')] += 1
                if cnt != ref:
                    return False
            return True
        
        # Try each divisor in ascending order, return the first that works
        for L in divisors:
            if can_use_length(L):
                return L
        
        # Fallback (shouldn't happen since L = n always works)
        return n

# Example usage:
# sol = Solution()
# print(sol.minAnagramLength("abba"))  # Output: 2
# print(sol.minAnagramLength("cdef"))  # Output: 4