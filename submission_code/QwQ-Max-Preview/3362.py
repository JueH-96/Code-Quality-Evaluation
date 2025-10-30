class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        max_distinct = len(set(nums))
        m = n * (n + 1) // 2
        K = (m - 1) // 2
        self.nums = nums
        self.max_num = max(nums) if nums else 0
        low, high = 1, max_distinct
        res = max_distinct
        while low <= high:
            mid = (low + high) // 2
            current = self.atMost(mid)
            if current > K:
                res = mid
                high = mid - 1
            else:
                low = mid + 1
        return res
    
    def atMost(self, d):
        max_num = self.max_num
        nums = self.nums
        freq = [0] * (max_num + 2)
        left = 0
        res = 0
        distinct = 0
        for right in range(len(nums)):
            x = nums[right]
            if freq[x] == 0:
                distinct += 1
            freq[x] += 1
            while distinct > d:
                y = nums[left]
                freq[y] -= 1
                if freq[y] == 0:
                    distinct -= 1
                left += 1
            res += right - left + 1
        return res