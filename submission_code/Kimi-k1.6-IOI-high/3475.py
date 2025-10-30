class Solution:
    def minOperations(self, nums: List[int]) -> int:
        arr = nums.copy()
        n = len(arr)
        count = 0
        
        for i in range(n - 2):
            if arr[i] == 0:
                arr[i] ^= 1
                arr[i+1] ^= 1
                arr[i+2] ^= 1
                count += 1
        
        for num in arr:
            if num == 0:
                return -1
        
        return count