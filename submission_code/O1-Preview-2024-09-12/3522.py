class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            valid = True
            for j in range(i, i + k - 1):
                if nums[j + 1] != nums[j] + 1:
                    valid = False
                    break
            if valid:
                power = nums[i + k - 1]
            else:
                power = -1
            results.append(power)
        return results