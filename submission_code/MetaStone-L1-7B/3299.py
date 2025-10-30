from collections import defaultdict

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        max_len = 0
        for x in freq:
            current_power = x
            max_m = 0
            while True:
                if current_power not in freq:
                    break
                if max_m > 0:
                    if freq[current_power] < 2:
                        break
                if max_m == 0:
                    if freq[current_power] < 1:
                        break
                max_m += 1
                current_power = current_power ** 2
            if max_m > 0:
                current_len = 2 * max_m + 1
                if current_len > max_len:
                    max_len = current_len
        return max_len