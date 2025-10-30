from collections import Counter

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_count = Counter()
        suffix_count = Counter(nums)
        result = []
        
        for i in range(n):
            prefix_count[nums[i]] += 1
            suffix_count[nums[i]] -= 1
            if suffix_count[nums[i]] == 0:
                del suffix_count[nums[i]]
            result.append(len(prefix_count) - len(suffix_count))
        
        return result