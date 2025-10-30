import math

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0

        # Calculate the maximum possible number of zeros (Z_max) in any dominant substring.
        # A substring with z zeros and o ones must satisfy o >= z^2.
        # The length of the substring is L = z + o.
        # Substituting o, we get L >= z + z^2.
        # The maximum possible length of a substring is n.
        # So, any dominant substring must satisfy z + z^2 <= n.
        # This inequality z^2 + z - n <= 0 holds for z <= (-1 + sqrt(1 + 4n)) / 2.
        # Z_max is the largest integer z that satisfies this.
        
        # We can find the largest integer z >= 0 such that z*z + z <= n using binary search.
        # The function f(z) = z*z + z is increasing for z >= 0.
        # We are looking for the maximum z such that f(z) <= n.
        # The range for z is from 0 up to roughly sqrt(n). A safe upper bound for binary search
        # could be int(sqrt(n)) + a small constant like 2 or 5.
        # If z = int(sqrt(n)) + 2, z*z approx n, z*z+z approx n+sqrt(n), which is likely > n.
        # So the search space [0, int(sqrt(n)) + 2] is sufficient.
        
        low, high = 0, int(math.sqrt(n)) + 2 
        max_zeros_in_dominant_substring = 0
        
        while low <= high:
            mid = (low + high) // 2
            
            # Check if mid*mid + mid <= n. Be careful about potential overflow if mid is large,
            # though with n <= 4e4, mid <= 202, mid*mid approx 40804, which is safe for standard integer types.
            
            val = mid * mid + mid
            
            if val <= n:
                # mid is a possible value for Z_max, try a larger one
                max_zeros_in_dominant_substring = mid
                low = mid + 1
            else:
                # mid is too large, Z_max must be smaller
                high = mid - 1

        # Now max_zeros_in_dominant_substring holds the largest integer Z_max such that Z_max^2 + Z_max <= n.
        # Any substring with `zeros` > Z_max cannot be dominant.
        # We can stop the inner loop early when the number of zeros exceeds Z_max.

        for i in range(n):
            zeros = 0
            ones = 0
            # Iterate j from i to n-1 to consider all substrings starting at i
            for j in range(i, n):
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                
                # Optimization: If the current number of zeros `zeros`
                # in s[i..j] is greater than the maximum possible `Z_max` for any dominant substring,
                # then s[i..j] cannot be dominant.
                # Furthermore, any longer substring s[i..j'] with j' > j will have
                # at least `zeros` zeros, so it also cannot be dominant.
                # Thus, we can stop considering `j'` for this starting `i`.
                if zeros > max_zeros_in_dominant_substring:
                    # Based on the calculation of Z_max, if zeros > Z_max,
                    # then zeros^2 + zeros > n.
                    # The length of s[i..j] is at most n.
                    # The number of ones is (length) - zeros <= n - zeros.
                    # We need ones >= zeros^2, i.e., (length) - zeros >= zeros^2,
                    # or (length) >= zeros^2 + zeros.
                    # Since (length) <= n < zeros^2 + zeros, this condition cannot be met.
                    # So, if zeros > max_zeros_in_dominant_substring, the substring is not dominant.
                    
                    # Break the inner loop and move to the next starting index i.
                    break 

                # Check if the current substring s[i..j] is dominant
                # This check is only performed for substrings where zeros <= max_zeros_in_dominant_substring
                # Note: zeros * zeros might overflow if zeros is extremely large, but zeros <= Z_max <= 202 here.
                # max zeros*zeros approx 202*202 = 40804, well within int limits.
                if ones >= zeros * zeros:
                    count += 1
                    
        return count