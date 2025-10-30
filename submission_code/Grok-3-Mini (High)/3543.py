class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        def count_at_most_k(char_target: str, max_count: int) -> int:
            n = len(s)
            left = 0
            count_char = 0
            ans = 0
            for right in range(n):
                if s[right] == char_target:
                    count_char += 1
                while count_char > max_count and left <= right:
                    if s[left] == char_target:
                        count_char -= 1
                    left += 1
                ans += (right - left + 1)
            return ans
        
        def count_both_at_most_k(max_count: int) -> int:
            n = len(s)
            left = 0
            count_zero = 0
            count_one = 0
            ans = 0
            for right in range(n):
                if s[right] == '0':
                    count_zero += 1
                else:
                    count_one += 1
                while (count_zero > max_count or count_one > max_count) and left <= right:
                    if s[left] == '0':
                        count_zero -= 1
                    else:
                        count_one -= 1
                    left += 1
                ans += (right - left + 1)
            return ans
        
        num_A = count_at_most_k('0', k)  # Number of substrings with at most k zeros
        num_B = count_at_most_k('1', k)  # Number of substrings with at most k ones
        num_both = count_both_at_most_k(k)  # Number of substrings with both at most k zeros and at most k ones
        return num_A + num_B - num_both