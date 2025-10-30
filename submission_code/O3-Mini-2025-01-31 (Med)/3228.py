from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # Let n be the even length of the arrays; we must keep exactly n/2 elements from each.
        # Our goal is to maximize the number of distinct elements in the union of the two selections.
        #
        # We have freedom to choose any n/2 elements from an array.
        # To “earn” a distinct number in the union, it must be picked in at least one array.
        # However, if a number appears in both arrays (a "common" number),
        # we only want to count it once in the union—and we can afford to pick it in only one array
        # if that helps free capacity in the other.
        #
        # So it is natural to break the distinct numbers into three parts:
        #   • only1 = numbers that appear in nums1 but not in nums2
        #   • only2 = numbers that appear in nums2 but not in nums1
        #   • common = numbers that appear in both.
        #
        # Note:
        # - Only numbers in only1 must be picked from nums1. Likewise, only2 only from nums2.
        # - In each array we can only pick at most half = n/2 distinct numbers.
        #   (Even though we choose n/2 elements, if there are duplicates that do not help the union,
        #    they don’t improve our answer.)
        #
        # Thus from nums1, we can secure at most min(|only1|, half) distinct numbers.
        # Similarly from nums2, at most min(|only2|, half).
        # For the common numbers, we can choose to assign each to one of the two arrays.
        # However, each array’s capacity for additional distinct elements
        # (beyond the numbers forced from only1 or only2) is limited by:
        #   capacity1 = max(0, half - min(|only1|, half))
        #   capacity2 = max(0, half - min(|only2|, half))
        # So at most capacity1 + capacity2 common numbers can be “used” to add to the union.
        #
        # Therefore, the maximum union distinct count is:
        #   min(|only1|, half) + min(|only2|, half) +
        #         min(|common|, (half - min(|only1|, half)) + (half - min(|only2|, half)))
        
        n = len(nums1)
        half = n // 2
        
        set1 = set(nums1)
        set2 = set(nums2)
        
        only1 = set1 - set2
        only2 = set2 - set1
        common = set1 & set2
        
        count_only1 = len(only1)
        count_only2 = len(only2)
        count_common = len(common)
        
        chosen_only1 = min(count_only1, half)
        chosen_only2 = min(count_only2, half)
        
        avail1 = half - chosen_only1  # additional capacity in nums1 for common numbers
        avail2 = half - chosen_only2  # additional capacity in nums2 for common numbers
        
        # We can use common numbers to fill the unused slots in either array—
        # each common number is counted only once in the union.
        common_can_pick = avail1 + avail2
        chosen_common = min(count_common, common_can_pick)
        
        return chosen_only1 + chosen_only2 + chosen_common

# The following code is for local testing and input parsing.
# It uses the helper method "maximumSetSize" of the Solution class.
if __name__ == "__main__":
    import sys
    data = sys.stdin.read().strip().split()
    
    if not data:
        sys.exit(0)
    
    # Input parsing:
    # One common way to supply input is to give the arrays on separate lines.
    # For example, input might be:
    #   1 2 1 2
    #   1 1 1 1
    # We'll assume that the first half of the tokens form nums1 and the rest form nums2.
    # Since the problem guarantees both arrays have equal (even) lengths, let m be len(data)//2.
    m = len(data) // 2
    nums1 = list(map(int, data[:m]))
    nums2 = list(map(int, data[m:2*m]))
    
    sol = Solution()
    result = sol.maximumSetSize(nums1, nums2)
    sys.stdout.write(str(result))