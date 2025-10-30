class Solution:
    def largestInteger(self, self, nums: List[int], k: int) -> int:
        counts = {}
        n = len(nums)
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            for num in subarray:
                if num not in counts:
                    counts[num] = 0
                counts[num] += 1
        
        almost_missing = []
        for num, count in counts.items():
            if count == 1:
                almost_missing.append(num)
        
        if not almost_missing:
            return -1
        else:
            return max(almost_missing)