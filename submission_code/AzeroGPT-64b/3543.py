class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        def substrings_count(s, target):
            left, right, count, total = 0, 0, 0, 0

            while right < len(s):
                if s[right] == target:
                    count += 1

                while count > k:
                    if s[left] == target:
                        count -= 1
                    left += 1

                total += right - left + 1
                right += 1

            return total

        zero_count = substrings_count(s, '0')
        one_count = substrings_count(s, '1')
        
        return zero_count + one_count - len(s)