from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # k is the number of elements kept from each array.
        k = len(nums1) // 2

        # Compute the distinct elements in each array.
        set1, set2 = set(nums1), set(nums2)
        d1, d2 = len(set1), len(set2)
        common = set1.intersection(set2)
        inter = len(common)
        
        # When we pick k elements from an array, if there are enough distinct ones,
        # we have the option to choose any k distinct numbers.
        # However, note that if an array has more than k distinct numbers,
        # we are forced (by the size restriction) to only use k of them.
        # Thus, define:
        x = min(k, d1)  # maximum distinct numbers chosen from nums1
        y = min(k, d2)  # maximum distinct numbers chosen from nums2
        
        # Now we split the distinct numbers in each array into two groups:
        #   Exclusive: those that appear only in that array.
        #   Common: those that appear in both arrays.
        #
        # For nums1 the count of exclusive elements is:
        exclusive1 = d1 - inter
        # For nums2:
        exclusive2 = d2 - inter
        
        # In an optimal strategy, when selecting x distinct numbers from nums1,
        # we can avoid picking too many common numbers if there are enough exclusive ones.
        # But if exclusive1 is less than x then we are forced to include some common numbers.
        common1 = max(0, x - exclusive1)
        # Similarly for nums2:
        common2 = max(0, y - exclusive2)
        
        # Our selections from common must come from the common set (of size inter).
        # We have control to assign the common elements to nums1 and nums2 picks.
        # If the total "need" for common numbers, common1 + common2, exceeds inter,
        # then even if we choose disjoint sets as much as possible,
        # there will be an unavoidable overlap of at least (common1 + common2 - inter)
        # common elements in both picks.
        overlap = max(0, common1 + common2 - inter)
        
        # The union of the chosen numbers is (x + y) minus the duplicated common ones.
        return x + y - overlap

# -------------------------
# Below are some tests to demonstrate the function.

if __name__ == '__main__':
    sol = Solution()
    
    # Example 1:
    nums1 = [1,2,1,2]
    nums2 = [1,1,1,1]
    # Expected output: 2
    print(sol.maximumSetSize(nums1, nums2))  # Output: 2
    
    # Example 2:
    nums1 = [1,2,3,4,5,6]
    nums2 = [2,3,2,3,2,3]
    # Expected output: 5
    print(sol.maximumSetSize(nums1, nums2))  # Output: 5
    
    # Example 3:
    nums1 = [1,1,2,2,3,3]
    nums2 = [4,4,5,5,6,6]
    # Expected output: 6
    print(sol.maximumSetSize(nums1, nums2))  # Output: 6

    # Additional test: identical arrays
    nums1 = [1,2,3,4]
    nums2 = [1,2,3,4]
    # One can choose disjoint subsets from the same pool.
    # For n=4, k=2, one optimal choice is [1,2] from the first array and [3,4] from the second.
    # Union = {1,2,3,4} so expected output: 4
    print(sol.maximumSetSize(nums1, nums2))  # Output: 4