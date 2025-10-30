class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Try both cases: keeping original last elements or swapping them
        def check(nums1, nums2, swapped_last):
            operations = 0 if not swapped_last else 1
            target1 = nums2[-1] if swapped_last else nums1[-1]
            target2 = nums1[-1] if swapped_last else nums2[-1]
            
            # Check each position except last
            for i in range(n-1):
                curr1, curr2 = nums1[i], nums2[i]
                
                # If current elements are greater than respective targets
                if curr1 > target1 or curr2 > target2:
                    # Try swapping at current position
                    if curr2 <= target1 and curr1 <= target2:
                        operations += 1
                    else:
                        return float('inf')
                        
            return operations
            
        # Try both possibilities
        ans1 = check(nums1, nums2, False)  # Keep last elements as is
        ans2 = check(nums1, nums2, True)   # Swap last elements
        
        final_ans = min(ans1, ans2)
        return final_ans if final_ans != float('inf') else -1