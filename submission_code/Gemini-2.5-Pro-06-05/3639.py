class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        """
        Determines if it's possible to make the nums array all zeros by applying a series of decrement operations.

        The core idea is to determine, for each index `i` in `nums`, the total number of
        times it can be decremented across all queries. If this number is at least `nums[i]`
        for every index, then it's possible to make the array a zero array.

        The problem states that for each query `[l, r]`, we can select *any subset* of
        indices in that range to decrement. This implies that the decision to decrement
        `nums[i]` is independent of the decision to decrement `nums[j]` for the same query
        (where `i, j` are in the range `[l, r]`).

        Therefore, for each index `i`, we can count the total number of queries that cover it.
        Let this be `coverage_count[i]`. This is the maximum number of times `nums[i]` can
        be decremented. To make `nums[i]` zero, we need exactly `nums[i]` decrements.
        This is possible if and only if `coverage_count[i] >= nums[i]`.

        The sequential nature of the queries and the non-negativity constraint are not
        an issue. Since we only need to decrement `nums[i]` a total of `nums[i]` times,
        and the initial value is `nums[i]`, the value at index `i` will never drop below
        zero during the process.

        To efficiently calculate `coverage_count` for all indices, we use a difference
        array (also known as a line sweep algorithm).
        """
        n = len(nums)
        
        # The difference array will help calculate the coverage count for each index
        # in a single pass. `diff[i]` stores the net change in the number of active
        # queries at index i.
        diff = [0] * (n + 1)
        
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n + 1:
                diff[r + 1] -= 1
        
        coverage_count = 0
        for i in range(n):
            # Update the coverage count for the current index i by adding the change.
            # This is equivalent to calculating the prefix sum of the diff array.
            coverage_count += diff[i]
            
            # The number of available decrements for nums[i] must be at least nums[i].
            if coverage_count < nums[i]:
                return False
                
        return True