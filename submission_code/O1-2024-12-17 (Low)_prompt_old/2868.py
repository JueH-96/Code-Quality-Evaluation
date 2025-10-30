class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        from collections import deque

        # Deques for tracking the minimum and maximum in the current window
        min_dq = deque()  # will store indices, maintaining values in non-decreasing order
        max_dq = deque()  # will store indices, maintaining values in non-increasing order

        l = 0
        ans = 0

        for r in range(len(nums)):
            # Insert current element's index in min_dq (remove bigger elements from the right)
            while min_dq and nums[min_dq[-1]] > nums[r]:
                min_dq.pop()
            min_dq.append(r)

            # Insert current element's index in max_dq (remove smaller elements from the right)
            while max_dq and nums[max_dq[-1]] < nums[r]:
                max_dq.pop()
            max_dq.append(r)

            # Shrink from the left if our condition (max-min <= 2) is violated
            while nums[max_dq[0]] - nums[min_dq[0]] > 2:
                # Move left pointer, removing it from deques if necessary
                if min_dq[0] == l:
                    min_dq.popleft()
                if max_dq[0] == l:
                    max_dq.popleft()
                l += 1

            # All subarrays ending at r and starting at or after l are valid
            ans += (r - l + 1)

        return ans