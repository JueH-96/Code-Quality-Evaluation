class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            valid = True
            for j in range(1, len(sub)):
                if sub[j] != sub[j-1] + 1:
                    valid = False
                    break
            results.append(sub[-1] if valid else -1)
        return results