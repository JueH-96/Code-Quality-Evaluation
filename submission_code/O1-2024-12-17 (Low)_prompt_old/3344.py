class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        """
        We want to remove exactly one point such that the maximum Manhattan distance
        among the remaining points is minimized.

        Manhattan distance between two points (x1, y1) and (x2, y2) is:
            |x1 - x2| + |y1 - y2|

        A standard trick to handle max Manhattan distance is to consider the values:
            T1 = x + y
            T2 = x - y
        The maximum Manhattan distance among a set of points is then:
            max( max(T1) - min(T1), max(T2) - min(T2) )

        After removing one point, we again look at:
            newMaxDist = max( (newMax(T1) - newMin(T1)), (newMax(T2) - newMin(T2)) )

        If the removed point contributes to a current extreme (e.g., the only point achieving max(T1)),
        we must "shift" to the second best extreme. Otherwise, that extreme remains unchanged.

        Steps:
          1) Precompute T1[i] = x_i + y_i, T2[i] = x_i - y_i for each point i.
          2) Find the largest (max1) and second largest (max2), smallest (min1) and second smallest (min2) values for T1 and T2. Also track how many points achieve each of those extremes (frequency).
          3) For each point i, compute the new extremes after removing i, in O(1) time:
             - If T1[i] = max1T1 and it's unique (freq=1), then newMaxT1 = max2T1, else newMaxT1 = max1T1.
               Similarly for newMinT1, max1T2, min1T2, etc.
             - The new maximum distance is max( newMaxT1 - newMinT1, newMaxT2 - newMinT2 ).
             - Track the minimum of these values over all i.
          4) Return that minimum.

        This runs in O(N) time, suitable for up to 10^5 points.
        """

        n = len(points)
        # Precompute t1[i] = x + y, t2[i] = x - y
        t1 = []
        t2 = []
        for x, y in points:
            t1.append(x + y)
            t2.append(x - y)

        # Helper function to retrieve: max1, freq_max1, max2, min1, freq_min1, min2
        def get_extremes(arr):
            max1 = -float('inf')
            max2 = -float('inf')
            min1 = float('inf')
            min2 = float('inf')
            freq_max1 = 0
            freq_min1 = 0

            for val in arr:
                # Update max side
                if val > max1:
                    max2 = max1
                    max1 = val
                    freq_max1 = 1
                elif val == max1:
                    freq_max1 += 1
                elif val > max2:
                    max2 = val

                # Update min side
                if val < min1:
                    min2 = min1
                    min1 = val
                    freq_min1 = 1
                elif val == min1:
                    freq_min1 += 1
                elif val < min2:
                    min2 = val

            return max1, freq_max1, max2, min1, freq_min1, min2

        # Get extremes for t1 and t2
        max1_t1, freq_max1_t1, max2_t1, min1_t1, freq_min1_t1, min2_t1 = get_extremes(t1)
        max1_t2, freq_max1_t2, max2_t2, min1_t2, freq_min1_t2, min2_t2 = get_extremes(t2)

        # Now, for each point i, determine the "new" extremes if that point is removed
        # and compute the new maximum distance. Track the minimum among all.
        ans = float('inf')
        for i in range(n):
            # Decide new max T1
            if t1[i] == max1_t1 and freq_max1_t1 == 1:
                new_max_t1 = max2_t1
            else:
                new_max_t1 = max1_t1

            # Decide new min T1
            if t1[i] == min1_t1 and freq_min1_t1 == 1:
                new_min_t1 = min2_t1
            else:
                new_min_t1 = min1_t1

            # Decide new max T2
            if t2[i] == max1_t2 and freq_max1_t2 == 1:
                new_max_t2 = max2_t2
            else:
                new_max_t2 = max1_t2

            # Decide new min T2
            if t2[i] == min1_t2 and freq_min1_t2 == 1:
                new_min_t2 = min2_t2
            else:
                new_min_t2 = min1_t2

            # Compute new maximum distance
            dist_t1 = new_max_t1 - new_min_t1
            dist_t2 = new_max_t2 - new_min_t2
            new_max_dist = max(dist_t1, dist_t2)

            # Update answer
            ans = min(ans, new_max_dist)

        return ans