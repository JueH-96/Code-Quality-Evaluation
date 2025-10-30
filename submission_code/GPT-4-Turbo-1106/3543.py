class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        def countSubstringsWithAtMostK(s, k):
            count = { '0': 0, '1': 0 }
            left = 0
            result = 0
            for right in range(len(s)):
                count[s[right]] += 1
                while count['0'] > k and count['1'] > k:
                    count[s[left]] -= 1
                    left += 1
                result += right - left + 1
            return result
        
        return countSubstringsWithAtMostK(s, k) - countSubstringsWithAtMostK(s, k - 1)