class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count_0, count_1 = [0] * (n + 1), [0] * (n + 1)
        
        # Precompute the prefix sums for 0's and 1's
        for i in range(1, n + 1):
            count_0[i] = count_0[i - 1] + (s[i - 1] == '0')
            count_1[i] = count_1[i - 1] + (s[i - 1] == '1')
        
        ans = 0
        
        # Iterate over all possible substrings
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                zeros = count_0[j] - count_0[i - 1]
                ones = count_1[j] - count_1[i - 1]
                
                # Check if the substring satisfies the k-constraint
                if zeros <= k and ones <= k:
                    ans += 1
                else:
                    break  # No need to check longer substrings starting from i
        
        return ans