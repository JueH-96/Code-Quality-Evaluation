class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        def solve(max1, max2, swapped_last):
            operations = 1 if swapped_last else 0
            
            for i in range(n - 1):
                # Check if current arrangement works
                can_keep = nums1[i] <= max1 and nums2[i] <= max2
                # Check if swapped arrangement works
                can_swap = nums2[i] <= max1 and nums1[i] <= max2
                
                if can_keep and can_swap:
                    # Both work, choose the one that doesn't require additional swap
                    continue
                elif can_swap and not can_keep:
                    # Must swap
                    operations += 1
                elif not can_keep and not can_swap:
                    # Neither works, impossible
                    return float('inf')
                # If can_keep and not can_swap, we keep (no additional operation)
            
            return operations
        
        # Scenario 1: Don't swap last elements
        # nums1 max should be nums1[n-1], nums2 max should be nums2[n-1]
        result1 = solve(nums1[n-1], nums2[n-1], False)
        
        # Scenario 2: Swap last elements
        # nums1 max should be nums2[n-1], nums2 max should be nums1[n-1]
        result2 = solve(nums2[n-1], nums1[n-1], True)
        
        min_ops = min(result1, result2)
        return min_ops if min_ops != float('inf') else -1