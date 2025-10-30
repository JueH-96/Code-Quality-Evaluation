class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # We want to answer queries of the form:
        #   "Among indices j where nums1[j] >= x and nums2[j] >= y,
        #    find the maximum of nums1[j] + nums2[j], or -1 if none."
        #
        # Because both n and len(queries) can each be up to 1e5, we need an O((n+Q) log(n+Q)) solution.
        # We'll use an "offline" approach:
        #
        # Steps:
        # 1) Pair up each index i into (nums1[i], nums2[i]) and compute sum[i] = nums1[i] + nums2[i].
        # 2) Sort these pairs in descending order by nums1, and if tied, descending by nums2.
        # 3) Each query (x, y) will also be processed in descending order of x.
        # 4) We need to efficiently find the maximum sum among pairs whose nums2 >= y.
        #    We'll use a Fenwick tree (BIT) or segment tree over nums2. Because nums2 can be up to 1e9,
        #    we must do coordinate compression on all possible y-values (both from nums2 and the queries).
        # 5) As we move through sorted queries by descending x, we will "activate" all pairs with nums1 >= x.
        #    For each activated pair, we update the Fenwick tree with the pair's nums2 coordinate → sum.
        # 6) Then to answer the query, we want the maximum sum for all indices coords >= query's y-coordinate.
        #    We'll do a Fenwick tree that can query a suffix maximum. This can be done by storing
        #    reversed indices in the Fenwick tree. 
        #
        # Implementation details below.

        import bisect

        n = len(nums1)

        # 1) Build array of points and their sums
        points = [(nums1[i], nums2[i]) for i in range(n)]
        # 2) Sort points by nums1 descending, tie-break by nums2 descending
        points.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Collect all y-values from nums2 and from queries so we can compress them
        all_y = list(nums2)
        for xq, yq in queries:
            all_y.append(yq)

        # Get unique sorted list of y-values
        unique_y = sorted(set(all_y))
        # For coordinate compression, we'll do: cY = bisect_left(unique_y, Y)
        # Then cY is in [0, len(unique_y)-1].

        # Fenwick (BIT) for maximum over reversed indices
        # We'll define:
        #   r = maxIndex - cY
        # so that a Fenwick prefix query up to r corresponds to a suffix query cY..end in normal order.
        #
        # Implementation of Fenwick for max:
        class FenwickMax:
            def __init__(self, size):
                self.size = size
                self.data = [-1_000_000_000_000_000_000] * (size + 1)  # very negative init

            def update(self, idx, val):
                # idx is 1-based in Fenwick logic
                while idx <= self.size:
                    if val > self.data[idx]:
                        self.data[idx] = val
                    idx += idx & -idx

            def query(self, idx):
                # returns max from 1..idx
                result = -1_000_000_000_000_000_000
                while idx > 0:
                    if self.data[idx] > result:
                        result = self.data[idx]
                    idx -= idx & -idx
                return result

        # Prepare Fenwick tree
        maxC = len(unique_y)  # number of compressed y-values
        fenwicks = FenwickMax(maxC)

        # We'll process queries in descending order of x, so store them with their index
        q_indexed = []
        for i, (xq, yq) in enumerate(queries):
            q_indexed.append((xq, yq, i))
        q_indexed.sort(key=lambda x: x[0], reverse=True)

        ans = [-1] * len(queries)

        # Now, we'll sweep through points in descending order of nums1,
        # and we'll keep a pointer p that we move as we lower the x threshold for queries.
        p = 0
        total_points = len(points)

        # We will maintain the maximum sums in fenwicks. Initially, none are "activated."
        # We'll proceed in descending x for queries.
        for xq, yq, idx in q_indexed:
            # Activate all points with nums1 >= xq
            while p < total_points and points[p][0] >= xq:
                # Compress the y for points[p][1]
                y_val = points[p][1]
                sum_val = points[p][0] + y_val
                cY = bisect.bisect_left(unique_y, y_val)  # 0-based
                # Convert to Fenwick index space in "reversed" manner
                # We'll store items at r = maxC - cY
                # But Fenwick is 1-based, so it's FenwicksIndex = (maxC - cY)
                r_index = maxC - cY
                # update Fenwicks
                fenwicks.update(r_index, sum_val)
                p += 1

            # Now the query "xq, yq"
            # We want maximum among those with nums2 >= yq.
            # That corresponds to compressed index cQ = bisect_left(unique_y, yq).
            cQ = bisect.bisect_left(unique_y, yq)
            if cQ == len(unique_y):
                # yq is larger than any existing y => no valid points
                ans[idx] = -1
            else:
                # Suffix from cQ..end => in Fenwicks, that is prefix up to (maxC - cQ).
                r_query = maxC - cQ
                best_val = fenwicks.query(r_query)
                # If best_val is still extremely negative, no valid activation
                if best_val < 0:
                    ans[idx] = -1
                else:
                    ans[idx] = best_val

        return ans