class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        def is_increasing(arr):
            for i in range(len(arr) - 1):
                if arr[i] >= arr[i+1]:
                    return False
            return True

        for a in range(n - 2 * k + 1):
            b = a + k
            sub1 = nums[a:a+k]
            sub2 = nums[b:b+k]
            
            if is_increasing(sub1) and is_increasing(sub2):
                return True
        return False