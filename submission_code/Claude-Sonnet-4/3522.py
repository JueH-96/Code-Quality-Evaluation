class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            # Check subarray from i to i+k-1
            is_consecutive_sorted = True
            
            # Check if all adjacent pairs are consecutive
            for j in range(i, i + k - 1):
                if nums[j + 1] != nums[j] + 1:
                    is_consecutive_sorted = False
                    break
            
            if is_consecutive_sorted:
                # Maximum element is the last element in sorted consecutive array
                results.append(nums[i + k - 1])
            else:
                results.append(-1)
        
        return results