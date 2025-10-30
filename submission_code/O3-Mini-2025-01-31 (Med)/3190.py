from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        ans = float('inf')
        
        # We plan to fix the final elements at index n-1 in both arrays.
        # There are two possibilities for index n-1:
        #   1. Do not swap at n-1.
        #      → final nums1[n-1] = nums1[n-1], final nums2[n-1] = nums2[n-1]
        #   2. Swap at n-1.
        #      → final nums1[n-1] = nums2[n-1], final nums2[n-1] = nums1[n-1]
        # We treat these choices as our two candidate configurations.
        for candidate in [0, 1]:
            if candidate == 0:
                target1 = nums1[n - 1]  # final value at index n-1 of nums1
                target2 = nums2[n - 1]  # final value at index n-1 of nums2
                op_count = 0
            else:
                target1 = nums2[n - 1]
                target2 = nums1[n - 1]
                op_count = 1  # we swapped at index n-1
            
            valid = True  # will remain True if a valid choice is found for all other indices
            
            # For each index i from 0 to n-2, decide independently whether to swap or not.
            # After the (possible) swap at index i, the final pair will be:
            #   • If we do not swap: (nums1[i], nums2[i])
            #   • If we swap: (nums2[i], nums1[i])
            # The two conditions that must be satisfied for each i are:
            #   nums1[i] <= target1  and  nums2[i] <= target2.
            # Equality is allowed because the last element is defined to be the maximum.
            for i in range(n - 1):
                no_swap_valid = (nums1[i] <= target1 and nums2[i] <= target2)
                swap_valid = (nums2[i] <= target1 and nums1[i] <= target2)
                if no_swap_valid:
                    # When the non-swap option is valid, choose it since it adds zero cost.
                    continue
                elif swap_valid:
                    # Otherwise, if only swapping keeps the value within bounds, do that.
                    op_count += 1
                else:
                    # No matter how we choose at index i, we cannot maintain
                    # the invariant that every value is <= the target at index n-1.
                    valid = False
                    break
            if valid:
                ans = min(ans, op_count)
        return ans if ans != float('inf') else -1


# The code below is provided for local testing and for online judges that require input/output handling.
def solve():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    pos = 0
    try:
        # Expect first token to be n.
        n = int(data[pos])
        pos += 1
    except:
        n = len(data) // 2
    # Next n tokens for nums1 and next n tokens for nums2.
    nums1 = list(map(int, data[pos: pos + n]))
    pos += n
    nums2 = list(map(int, data[pos: pos + n]))
    sol = Solution()
    result = sol.minOperations(nums1, nums2)
    sys.stdout.write(str(result))
    
if __name__ == "__main__":
    solve()
    
# For class usage:
if __name__ == "__main__" and False:
    # Sample test cases -- these lines will not run during online judging.
    sol = Solution()
    # Example 1: Expected output 1
    print(sol.minOperations([1, 2, 7], [4, 5, 3]))
    # Example 2: Expected output 2
    print(sol.minOperations([2, 3, 4, 5, 9], [8, 8, 4, 4, 4]))
    # Example 3: Expected output -1
    print(sol.minOperations([1, 5, 4], [2, 5, 3]))