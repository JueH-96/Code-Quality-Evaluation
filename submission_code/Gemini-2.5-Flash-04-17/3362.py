from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        
        # The median is the element at index (total_subarrays - 1) // 2 in the sorted uniqueness array.
        # We are looking for the smallest value 'v' such that the count of subarrays
        # with distinct count <= v is at least (total_subarrays - 1) // 2 + 1.
        # (total_subarrays - 1) // 2 + 1 simplifies to (total_subarrays + 1) // 2.
        target_count = (total_subarrays + 1) // 2

        def count_le(k: int) -> int:
            """
            Counts the number of subarrays with at most k distinct elements.
            Uses a sliding window [left, right].
            For a fixed right, we find the smallest left such that nums[left..right]
            has <= k distinct elements. All subarrays nums[i..right] with i >= left
            satisfy the condition. There are (right - left + 1) such subarrays.
            """
            count = 0
            left = 0
            # Use an array for frequency map since value range is limited (1 <= nums[i] <= 10^5)
            max_val = 100000 # Based on constraint 1 <= nums[i] <= 10^5
            freq = [0] * (max_val + 1) # Indices 0 to max_val
            distinct_count = 0

            for right in range(n):
                # Add nums[right] to the window
                num_right = nums[right]
                if freq[num_right] == 0:
                    distinct_count += 1
                freq[num_right] += 1

                # Shrink the window from the left if distinct count exceeds k
                while distinct_count > k:
                    num_left = nums[left]
                    freq[num_left] -= 1
                    if freq[num_left] == 0:
                        distinct_count -= 1
                    left += 1

                # For the current `right`, the window is `nums[left..right]`.
                # Any subarray ending at `right` with starting index >= `left`
                # also has at most k distinct elements.
                # The possible starting indices are `left, left + 1, ..., right`.
                # The number of such subarrays is `right - left + 1`.
                # Add the count of valid subarrays ending at `right`.
                count += (right - left + 1)

            return count

        # Binary search for the median value in the range [1, n]
        low = 1 # Minimum possible distinct count is 1
        high = n # Maximum possible distinct count is n
        ans = n # Initialize with a value in the possible range (will be updated)

        # We are looking for the smallest value 'v' such that count_le(v) >= target_count.
        # This is a standard binary search problem: find the first value that satisfies a condition.
        while low <= high:
            mid = (low + high) // 2
            if count_le(mid) >= target_count:
                # mid satisfies the condition (count is sufficient).
                # mid could be the answer, or maybe a smaller value works.
                ans = mid
                high = mid - 1 # Try searching in the lower half
            else:
                # mid does not satisfy the condition (count is too low).
                # mid is too small, need a larger value.
                low = mid + 1 # Try searching in the upper half

        return ans