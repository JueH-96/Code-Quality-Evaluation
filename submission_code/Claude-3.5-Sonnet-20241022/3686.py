class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        
        def is_prefix(arr1, arr2):
            # Check if arr1 is a prefix of arr2
            if len(arr1) > len(arr2):
                return False
            return arr1 == arr2[:len(arr1)]
        
        def check_split(i, j):
            # Get the three subarrays
            nums1 = nums[:i]
            nums2 = nums[i:j]
            nums3 = nums[j:]
            
            # Check if nums1 is prefix of nums2 OR nums2 is prefix of nums3
            return is_prefix(nums1, nums2) or is_prefix(nums2, nums3)
        
        count = 0
        # Try all possible splits
        # i marks the end of nums1, j marks the end of nums2
        for i in range(1, n-1):  # Leave at least one element for nums2 and nums3
            for j in range(i+1, n):  # Leave at least one element for nums3
                if check_split(i, j):
                    count += 1
                    
        return count