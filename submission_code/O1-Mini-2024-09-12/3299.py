from collections import Counter
class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        max_length = 0
        for x in freq:
            if x == 1:
                if freq[x] % 2 == 1:
                    max_length = max(max_length, freq[x])
                else:
                    max_length = max(max_length, freq[x] - 1)
            else:
                length = 1
                current_x = x
                while True:
                    x2 = current_x * current_x
                    if x2 in freq and freq[current_x] >= 2:
                        length += 2
                        current_x = x2
                    else:
                        break
                max_length = max(max_length, length)
        return max_length