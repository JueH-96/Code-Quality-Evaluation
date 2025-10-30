from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # 1. Sort nums in non-decreasing order
        nums.sort()
        n = len(nums)

        # 2. Calculate the initial sum of all elements.
        # Python handles large integers automatically, so a standard int is sufficient.
        current_sum = sum(nums)

        # 3. Iterate from the largest element backwards.
        # We consider potential polygon sides formed by the first i+1 elements
        # from the sorted array: nums[0], ..., nums[i].
        # The largest element in this subset is nums[i].
        # For a polygon with k sides (k >= 3), the sum of the k-1 smallest sides must be strictly greater than the largest side.
        # We need at least 3 elements to form a polygon, so the number of elements i+1 must be >= 3, meaning the index i must be at least 2.
        # The loop iterates index i from n-1 down to 2 (inclusive).
        for i in range(n - 1, 1, -1):
            # At the start of this iteration (for index i), current_sum holds the sum
            # of the elements nums[0] through nums[i].
            
            # The largest element in the current subset (nums[0]...nums[i]) is nums[i].
            # The sum of the other i elements is current_sum - nums[i] (which is sum(nums[0]...nums[i-1])).
            
            # Check the polygon condition: sum of smaller sides > largest side
            if current_sum - nums[i] > nums[i]:
                # If the condition holds for the set nums[0]...nums[i], we found a valid polygon.
                # The perimeter is the sum of these sides, which is current_sum.
                # Since we are iterating from larger potential subsets (n elements down to 3,
                # using the initial elements of the sorted array), the first valid one found
                # will have the largest perimeter among such subsets.
                # The logic guarantees this is the overall largest possible perimeter.
                return current_sum
            else:
                # If the condition does not hold for nums[0]...nums[i], it means nums[i] is too large
                # relative to the sum of the smaller elements (nums[0]...nums[i-1]) to form a polygon
                # with nums[i] as the longest side.
                # As reasoned in the thought block, this implies nums[i] cannot be part of any largest
                # perimeter polygon. We discard nums[i] from consideration for forming the largest polygon.
                # For the next iteration (i-1), we will consider the set nums[0]...nums[i-1].
                # Update current_sum to be the sum of nums[0]...nums[i-1] for the next iteration.
                current_sum -= nums[i]

        # 4. If the loop finishes (i goes down to 1), it means no subset of size >= 3
        # considered (nums[0]...nums[i] for i >= 2) formed a valid polygon.
        # Return -1.
        return -1