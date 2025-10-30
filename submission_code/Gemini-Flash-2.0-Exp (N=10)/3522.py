class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            sorted_subarray = sorted(subarray)
            is_consecutive = True
            if len(set(subarray)) != k:
                is_consecutive = False
            else:
                for j in range(k - 1):
                    if sorted_subarray[j+1] - sorted_subarray[j] != 1:
                        is_consecutive = False
                        break
            if is_consecutive:
                results.append(max(subarray))
            else:
                results.append(-1)
        return results