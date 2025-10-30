import math

class Solution:
    def minimumOperations(self, num: str) -> int:
        def max_length_for_pair(left_char, right_char):
            n = len(num)
            max_len = -1  # Invalid value to indicate no valid subsequence found
            for i in range(n):
                if num[i] == left_char:
                    # Check if there is a j > i with num[j] == right_char
                    has_right = False
                    for j in range(i + 1, n):
                        if num[j] == right_char:
                            has_right = True
                            break
                    if has_right:
                        max_len = max(max_len, i + 2)
            return max_len
        
        # Compute the maximum length for each possible ending pair
        len_00 = max_length_for_pair('0', '0')
        len_25 = max_length_for_pair('2', '5')
        len_50 = max_length_for_pair('5', '0')
        len_75 = max_length_for_pair('7', '5')
        
        # Find the maximum length among all suffix endings
        max_suffix_len = max(len_00, len_25, len_50, len_75)
        
        if max_suffix_len != -1:
            # If a valid suffix ending is possible, use that length
            max_special_len = max_suffix_len
        else:
            # Otherwise, use the count of zeros to form 0
            max_special_len = num.count('0')
        
        # Minimum operations is the total length minus the length of the longest special subsequence
        min_ops = len(num) - max_special_len
        return min_ops