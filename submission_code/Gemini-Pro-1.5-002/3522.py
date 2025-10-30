class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            is_consecutive = True
            for j in range(1, k):
                if subarray[j] != subarray[j-1] + 1:
                    if subarray[j] != subarray[j -1]:
                        is_consecutive = False
                        break
            if is_consecutive:
                
                is_sorted = all(subarray[j] <= subarray[j+1] for j in range(k-1))
                if is_sorted:
                    results.append(subarray[-1])
                
                else:
                    results.append(-1)
            else:
                results.append(-1)
        return results