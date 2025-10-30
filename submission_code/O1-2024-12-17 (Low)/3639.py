class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        # We want to check if each element in nums can be decremented enough times to become zero.
        # For each index i, we can decrement it at most as many times as the number of queries covering i.
        # Hence, we compute the coverage: coverage[i] = number of queries covering index i.
        # We then compare coverage[i] with nums[i]. If nums[i] <= coverage[i] for all i, return True; otherwise, False.

        n = len(nums)
        coverage_diff = [0] * (n + 1)

        # Build a difference array to count coverages of each index
        for l, r in queries:
            coverage_diff[l] += 1
            if r + 1 < n:
                coverage_diff[r + 1] -= 1

        # Convert difference array to the actual coverage array
        coverage = [0] * n
        running_sum = 0
        for i in range(n):
            running_sum += coverage_diff[i]
            coverage[i] = running_sum

        # Check if each element can be decremented enough times
        for i in range(n):
            if nums[i] > coverage[i]:
                return False

        return True