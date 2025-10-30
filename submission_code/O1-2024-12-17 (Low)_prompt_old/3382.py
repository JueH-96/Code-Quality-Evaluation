class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        """
        We want to count the number of subarrays where the first element, the last element,
        and the maximum element of the subarray are all the same. Equivalently:

        A subarray [i..j] is valid if:
          1) nums[i] == nums[j], call this value v
          2) No element in nums[i..j] is greater than v

        An efficient way to count:
          • For each index i, let v = nums[i].
          • We only form subarrays [i..j] where nums[j] == v and no element in between
            exceeds v. That means j < next_greater[i], where next_greater[i] is the index
            of the next position to the right with a strictly larger value than v (or len(nums)
            if none).
          • We gather all indices of each value in a dictionary: value -> sorted list of indices.
          • For each i, we binary search in that list of v to count how many indices j lie in
            [i, next_greater[i]) (i.e., j >= i and j < next_greater[i]).

        Steps:
          1. Compute next_greater using a monotonic stack (right-to-left).
          2. Build the dictionary from the values to the sorted indices.
          3. For each i, do a range count in dictionary[nums[i]] for indices in [i, next_greater[i]).
          4. Sum all counts.

        This yields the total number of valid subarrays.
        """

        n = len(nums)
        # Step 1: Compute next_greater for each position
        # next_greater[i] will be the smallest index > i where nums[index] > nums[i].
        # If none, we'll store n.
        next_greater = [n]*n
        stack = []
        # We'll process from right to left
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            next_greater[i] = stack[-1] if stack else n
            stack.append(i)

        # Step 2: Build dictionary value -> sorted list of indices
        from collections import defaultdict
        positions = defaultdict(list)
        for idx, val in enumerate(nums):
            positions[val].append(idx)

        # Step 3: For each i, count how many indices j in positions[nums[i]]
        # that lie within [i, next_greater[i])
        # We'll use bisect in Python
        import bisect

        ans = 0
        for i in range(n):
            val = nums[i]
            left_bound = i
            right_bound = next_greater[i]  # j must be < right_bound

            idx_list = positions[val]
            # Count how many in idx_list are >= left_bound and < right_bound
            left_idx = bisect.bisect_left(idx_list, left_bound)
            right_idx = bisect.bisect_left(idx_list, right_bound)
            ans += (right_idx - left_idx)

        return ans