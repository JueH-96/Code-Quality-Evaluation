from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        # Explanation:
        # An operation is merging two adjacent numbers by taking their bitwise AND.
        # Notice that if we merge a contiguous block of numbers, the merged result is the
        # AND of all numbers in that block.
        # Also, note that the bitwise OR of the final array is the OR of the merged results
        # from each contiguous block.
        #
        # Key observation: Consider a single bit position b.
        # For a contiguous block (segment) whose merged value (the AND of all numbers)
        # to have bit b == 0, it is necessary that at least one number in that segment
        # has that bit 0.
        #
        # We are allowed to perform at most k operations. Since initially there are n segments
        # (each single element), performing m operations will partition the array into (n - m)
        # segments. Because m <= k, the maximum number of segments we can have is at least n - k.
        # In the final partition we are free to choose any segmentation as long as the number of segments
        # p satisfies p = n - m with m <= k, i.e. p >= n - k.
        #
        # Now, observe that if we want every segment to have bit b cleared (i.e. AND result at that bit = 0),
        # then every segment must contain at least one number having a 0 at bit b.
        # If you think about it, the maximum number of segments you can get with that property is limited
        # by the number of elements in nums that are 0 at that bit (since you can always “cut” right at
        # a position which is 0 to isolate that element in its own segment).
        # Therefore, in order for it to be possible to “force” bit b to become 0 in every merged result,
        # you must have that:
        #
        #   (# of nums with bit b == 0) >= (number of segments needed) >= (n - k)
        #
        # If that condition does NOT hold, then no matter how you merge adjacent elements,
        # at least one segment will come out with bit b = 1, and so bit b will appear in the OR.
        #
        # Thus for each bit position b (from 0 to 30 because 0 <= nums[i] < 2^30) we count the number of
        # elements that have that bit equal to 0. Let countZero_b be that number.
        # If countZero_b >= (n - k) then it is possible to merge (choose a segmentation) so that every segment
        # gets at least one number with a 0 at b and so b can be “cleared” from the final OR.
        # Otherwise, b must remain in the final OR.
        #
        # Hence, the answer is the bitwise OR of exactly those bits b for which countZero_b < (n - k).
        
        n_val = len(nums)
        ans = 0
        for b in range(31):
            cnt_zero = 0
            mask = 1 << b
            for num in nums:
                if (num & mask) == 0:
                    cnt_zero += 1
            # Check if we can “clear” bit b in every segment.
            if cnt_zero < n_val - k:
                ans |= mask
        return ans


# --- Below is a helper main program to run the solution. ---
# It is useful when running this script independently.
if __name__ == '__main__':
    import sys
    data = sys.stdin.read().split()
    if not data:
        # In case there is no input, run the examples.
        sol = Solution()
        # Example 1:
        print(sol.minOrAfterOperations([3,5,3,2,7], 2))   # Expected output: 3
        # Example 2:
        print(sol.minOrAfterOperations([7,3,15,14,2,8], 4)) # Expected output: 2
        # Example 3:
        print(sol.minOrAfterOperations([10,7,10,3,9,14,9,4], 1)) # Expected output: 15
    else:
        # Input format: first integer n, then integer k, and then n numbers
        it = iter(data)
        n = int(next(it))
        k = int(next(it))
        nums = [int(next(it)) for _ in range(n)]
        sol = Solution()
        sys.stdout.write(str(sol.minOrAfterOperations(nums, k)))