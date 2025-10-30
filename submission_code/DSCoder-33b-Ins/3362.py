class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        n = len(nums)
        distinct_counts = []
        count = defaultdict(int)
        j = 0
        for i in range(n):
            while j < n and count[nums[j]] == 0:
                count[nums[j]] += 1
                j += 1
            distinct_counts.append(j - i)
            count[nums[i]] -= 1
        uniqueness_array = []
        for i in range(n):
            for j in range(i, n):
                uniqueness_array.append(distinct_counts[j])
        uniqueness_array.sort()
        return uniqueness_array[len(uniqueness_array) // 2]