from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        distinct_nums = sorted(freq.keys())
        ans = 0
        for num in distinct_nums:
            if num == 1:
                n = freq[1]
                chain_length = n if n % 2 == 1 else n - 1
                if chain_length > ans:
                    ans = chain_length
            else:
                chain_length = 1
                current = num
                while True:
                    if freq[current] < 2:
                        break
                    next_val = current * current
                    if next_val > 10**9:
                        break
                    if next_val not in freq:
                        break
                    chain_length += 2
                    current = next_val
                if chain_length > ans:
                    ans = chain_length
        return ans