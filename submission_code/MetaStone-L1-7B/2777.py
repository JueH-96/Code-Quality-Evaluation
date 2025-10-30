class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_counts = []
        current_set = set()
        for num in nums:
            current_set.add(num)
            prefix_counts.append(len(current_set))
        
        suffix_counts = [0] * n
        current_set = set()
        for i in range(n - 2, -1, -1):
            current_set.add(nums[i + 1])
            suffix_counts[i] = len(current_set)
        
        result = []
        for i in range(n):
            diff = prefix_counts[i] - suffix_counts[i]
            result.append(diff)
        return result