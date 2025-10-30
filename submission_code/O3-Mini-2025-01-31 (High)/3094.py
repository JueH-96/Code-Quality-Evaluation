from typing import List
import collections

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Count the frequency of each number.
        freq = collections.Counter(nums)
        total_ops = 0
        
        # For each distinct number, we need to partition its frequency count
        # into a sum of 2's and 3's (each representing one operation).
        # That is, for a frequency f we need non-negative integers x and y such that:
        #       2*x + 3*y = f
        # and we want to minimize the number of operations x+y.
        #
        # We observe that every f >= 2 can be written as a sum of 2's and 3's,
        # except when f == 1. So if any frequency is 1, the answer is -1.
        #
        # A neat closed-form solution can be derived via case analysis:
        # Let r = f % 3.
        # • If r == 0   : Best solution is to use f//3 operations (all triple removals).
        # • If r == 1   : Then f must be at least 4; we can use one less triple removal,
        #                 converting the remaining into two double removals:
        #                 ops = (f-4)//3 + 2.
        # • If r == 2   : Then ops = (f-2)//3 + 1.
        #
        for count in freq.values():
            if count == 1:
                return -1  # Not possible to remove a group of single element.
            if count % 3 == 0:
                total_ops += count // 3
            elif count % 3 == 1:
                # f must be at least 4 if remainder is 1 (f==1 is caught above)
                total_ops += (count - 4) // 3 + 2
            else:  # count % 3 == 2
                total_ops += (count - 2) // 3 + 1

        return total_ops


# Example usage and simple testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    nums1 = [2,3,3,2,2,4,2,3,4]
    print(sol.minOperations(nums1))  # Expected output: 4
    # Example 2:
    nums2 = [2,1,2,2,3,3]
    print(sol.minOperations(nums2))  # Expected output: -1