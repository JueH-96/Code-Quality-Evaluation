class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        """
        We need the smallest subset of queries that can still reduce nums to all zeros.
        Each query covers its [l, r] range exactly once (if chosen), providing
        '1 unit' of decrement to each index in that range. Each index i must be covered
        at least nums[i] times in total. We want to remove as many queries as possible,
        so equivalently we want the minimal number of queries that cover each i at
        least nums[i] times.

        Key idea:
          • Sort queries by their left endpoint.
          • Sweep from i = 0 to len(nums)-1 and maintain how many times index i
            is covered so far (coverage[i]).
          • If coverage[i] < nums[i], we must pick more queries that include i.
            We choose the queries that extend furthest to the right first (a standard
            greedy approach), so that future indices get covered “for free”.
          • Use a difference array 'diff' to track coverage increments. For each
            chosen query [l, r]: diff[l] += 1, diff[r+1] -= 1 (if r+1 in range),
            so that the coverage of every index in [l, r] is increased by 1.
          • Maintain a max-heap (by r) of all queries whose l ≤ i (and which have not
            been used yet). If r < i for the top of the heap, discard it (useless now).
            While coverage[i] < nums[i], pop from the heap and "use" that query,
            updating diff and coverage accordingly.
          • If at any point coverage[i] < nums[i] but the heap is empty, it's impossible.

        Steps:
          1. Sort queries by l ascending. 
          2. Keep a pointer p over queries to move them into a heap (max by r).
          3. Iterate i from 0..n-1:
             a) coverage[i] = coverage[i-1] + diff[i] (running sum)
             b) Add all queries with l ≤ i to the heap. Remove from top if r < i.
             c) While coverage[i] < nums[i], pop the top (largest r) from heap:
                - If none in heap, return -1
                - Use that query: increment coverage[i] by 1, do diff[l]++, diff[r+1]--,
                  and increment "used_queries" count.
          4. If we finish successfully, answer = total_queries - used_queries.

        Complexity: O((n + m) * log(m)) which is acceptable for n,m up to 10^5.
        """

        import heapq

        n = len(nums)
        m = len(queries)

        # Edge case: if all nums are zero, we can remove all queries.
        if all(x == 0 for x in nums):
            return m  # no queries needed at all

        # Sort queries by their left endpoint
        queries.sort(key=lambda x: x[0])  # each query = [l, r]

        # Difference array to track coverage increments.
        # diff[i] means coverage[i] changes by diff[i] relative to coverage[i-1].
        diff = [0] * (n + 1)

        # We'll keep a running 'coverage' as we move left to right.
        coverage = 0

        used_queries = 0
        pq = []   # max-heap (store as (-r, l) so largest r is on top)
        p = 0     # pointer into queries

        for i in range(n):
            # First apply any pending difference array increments for i
            coverage += diff[i]

            # If there are queries whose left endpoint <= i, push them into the heap
            while p < m and queries[p][0] <= i:
                l, r = queries[p]
                if r >= i:
                    # only useful if r >= i (it can cover i)
                    heapq.heappush(pq, (-r, l))
                p += 1

            # Also pop from heap any intervals that can no longer cover i (their r < i)
            while pq and -pq[0][0] < i:
                heapq.heappop(pq)

            # If coverage at i is not enough, pick intervals with the largest r
            while coverage < nums[i]:
                if not pq:
                    return -1
                # Pop the interval with the largest r
                neg_r, l = heapq.heappop(pq)
                r = -neg_r

                # We'll use this interval to increase coverage for [l..r] by 1
                used_queries += 1
                coverage += 1  # immediate effect on index i (since l <= i <= r)
                diff[l] += 1
                if r + 1 < n:
                    diff[r + 1] -= 1

        # If we completed covering all i, the minimal set we used is used_queries
        return m - used_queries