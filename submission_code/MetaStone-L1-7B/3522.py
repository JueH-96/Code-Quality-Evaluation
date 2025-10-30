class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        result = []
        n = len(nums)
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            is_consecutive = True
            for j in range(1, k):
                if sub[j] != sub[j-1] + 1:
                    is_consecutive = False
                    break
            if is_consecutive:
                result.append(sub[-1])
            else:
                result.append(-1)
        return result