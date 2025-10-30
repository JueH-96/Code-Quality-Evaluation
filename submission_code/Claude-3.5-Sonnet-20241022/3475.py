class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return -1
            
        # Try both possibilities for first element
        min_ops = float('inf')
        
        # Try original array
        nums1 = nums.copy()
        ops1 = 0
        for i in range(n-2):
            if nums1[i] == 0:
                # Flip next 3 elements
                ops1 += 1
                for j in range(i, min(i+3, n)):
                    nums1[j] = 1 - nums1[j]
                    
        # Check if all 1s
        if all(x == 1 for x in nums1):
            min_ops = ops1
            
        # Try flipping first 3 elements
        nums2 = nums.copy()
        ops2 = 1
        for j in range(3):
            nums2[j] = 1 - nums2[j]
            
        for i in range(n-2):
            if nums2[i] == 0:
                ops2 += 1
                for j in range(i, min(i+3, n)):
                    nums2[j] = 1 - nums2[j]
                    
        if all(x == 1 for x in nums2):
            min_ops = min(min_ops, ops2)
            
        return min_ops if min_ops != float('inf') else -1