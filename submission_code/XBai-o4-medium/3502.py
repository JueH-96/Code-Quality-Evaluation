class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        r = k - 1
        
        def count_valid_substrings(max_repeats):
            count = [0] * 26
            left = 0
            res = 0
            for right in range(n):
                c = s[right]
                idx = ord(c) - ord('a')
                count[idx] += 1
                
                while count[idx] > max_repeats:
                    left_c = s[left]
                    idx_left = ord(left_c) - ord('a')
                    count[idx_left] -= 1
                    left += 1
                res += right - left + 1
            return res
        
        valid = count_valid_substrings(r)
        return total - valid