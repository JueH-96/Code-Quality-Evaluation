class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        import heapq

        n = len(nums)
        m = len(queries)

        # Sort queries by their starting index
        queries.sort(key=lambda x: x[0])

        # A max-heap (using negative r) for potential intervals we might choose
        potential = []
        # A min-heap (storing r) for intervals we have chosen (active coverage)
        active = []

        chosen = 0   # Number of queries actually used
        p = 0        # Pointer to iterate over sorted queries

        for i in range(n):
            # Remove from 'active' all intervals that no longer cover index i
            while active and active[0] < i:
                heapq.heappop(active)

            coverage_i = len(active)  # How many chosen queries cover i

            # Add to 'potential' all queries whose left <= i
            while p < m and queries[p][0] <= i:
                # Push negative of r so we pop the largest r first from 'potential'
                heapq.heappush(potential, -queries[p][1])
                p += 1

            # If current coverage is less than needed at index i, pick more intervals
            needed = nums[i]
            if coverage_i < needed:
                short = needed - coverage_i
                while short > 0:
                    if not potential:
                        return -1  # Not enough queries to achieve needed coverage
                    largest_r = -heapq.heappop(potential)
                    # Now this query is chosen, so it covers i..largest_r
                    heapq.heappush(active, largest_r)
                    chosen += 1
                    short -= 1

        # We successfully covered all indices
        return m - chosen