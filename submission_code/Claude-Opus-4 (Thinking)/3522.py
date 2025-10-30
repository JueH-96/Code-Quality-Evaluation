class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        
        # consecutive[i] = number of consecutive increasing elements starting from position i
        consecutive = [1] * n
        for i in range(n - 2, -1, -1):
            if nums[i] + 1 == nums[i + 1]:
                consecutive[i] = consecutive[i + 1] + 1
        
        results = []
        for i in range(n - k + 1):
            if consecutive[i] >= k:
                # The subarray starting at i has k consecutive increasing elements
                # The maximum is the last element
                results.append(nums[i + k - 1])
            else:
                results.append(-1)
        
        return results