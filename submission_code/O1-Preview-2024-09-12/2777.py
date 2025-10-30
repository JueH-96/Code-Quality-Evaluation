class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_counts = []
        seen_prefix = set()
        for num in nums:
            seen_prefix.add(num)
            prefix_counts.append(len(seen_prefix))
        
        suffix_counts = [0]*n
        seen_suffix = set()
        for i in range(n-1, -1, -1):
            if i == n - 1:
                suffix_counts[i] = 0
            else:
                suffix_counts[i] = suffix_counts[i+1]
                if nums[i+1] not in seen_suffix:
                    seen_suffix.add(nums[i+1])
                    suffix_counts[i] += 1
        return [prefix_counts[i] - suffix_counts[i] for i in range(n)]