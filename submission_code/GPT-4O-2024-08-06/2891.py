class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        # Dictionary to store the frequency of each number
        frequency = defaultdict(int)

        # Iterate over each number in nums
        for num in nums:
            # Increment the frequency for each number in the range [num-k, num+k]
            for i in range(num - k, num + k + 1):
                frequency[i] += 1

        # The maximum frequency found will be the maximum beauty
        max_beauty = max(frequency.values())

        return max_beauty