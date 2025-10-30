class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)

        s_vals = []
        d_vals = []

        for i in range(n):
            x_i, y_i = points[i]
            s_i = x_i + y_i
            d_i = x_i - y_i
            s_vals.append((s_i, i))
            d_vals.append((d_i, i))

        # Initialize top 2 max and min for s_i and d_i
        max_s_vals = [(-float('inf'), -1), (-float('inf'), -1)]
        min_s_vals = [(float('inf'), -1), (float('inf'), -1)]
        max_d_vals = [(-float('inf'), -1), (-float('inf'), -1)]
        min_d_vals = [(float('inf'), -1), (float('inf'), -1)]

        for i in range(n):
            s_i, idx = s_vals[i]
            d_i, idx = d_vals[i]

            # Update max_s_vals
            if s_i > max_s_vals[0][0]:
                max_s_vals[1] = max_s_vals[0]
                max_s_vals[0] = (s_i, idx)
            elif s_i > max_s_vals[1][0]:
                max_s_vals[1] = (s_i, idx)

            # Update min_s_vals
            if s_i < min_s_vals[0][0]:
                min_s_vals[1] = min_s_vals[0]
                min_s_vals[0] = (s_i, idx)
            elif s_i < min_s_vals[1][0]:
                min_s_vals[1] = (s_i, idx)

            # Update max_d_vals
            if d_i > max_d_vals[0][0]:
                max_d_vals[1] = max_d_vals[0]
                max_d_vals[0] = (d_i, idx)
            elif d_i > max_d_vals[1][0]:
                max_d_vals[1] = (d_i, idx)

            # Update min_d_vals
            if d_i < min_d_vals[0][0]:
                min_d_vals[1] = min_d_vals[0]
                min_d_vals[0] = (d_i, idx)
            elif d_i < min_d_vals[1][0]:
                min_d_vals[1] = (d_i, idx)

        # Compute initial maximum distance
        initial_dist = max(
            max_s_vals[0][0] - min_s_vals[0][0],
            max_d_vals[0][0] - min_d_vals[0][0]
        )

        # Collect candidate indices to remove
        candidate_indices = set()
        candidate_indices.add(max_s_vals[0][1])
        candidate_indices.add(max_s_vals[1][1])
        candidate_indices.add(min_s_vals[0][1])
        candidate_indices.add(min_s_vals[1][1])
        candidate_indices.add(max_d_vals[0][1])
        candidate_indices.add(max_d_vals[1][1])
        candidate_indices.add(min_d_vals[0][1])
        candidate_indices.add(min_d_vals[1][1])

        # Remove invalid indices (-1)
        candidate_indices.discard(-1)

        min_dist = float('inf')

        for idx in candidate_indices:
            # For s_i
            # Max s_i after removing idx
            if idx != max_s_vals[0][1]:
                max_s = max_s_vals[0][0]
            else:
                max_s = max_s_vals[1][0] if max_s_vals[1][1] != -1 else max_s_vals[0][0]

            # Min s_i after removing idx
            if idx != min_s_vals[0][1]:
                min_s = min_s_vals[0][0]
            else:
                min_s = min_s_vals[1][0] if min_s_vals[1][1] != -1 else min_s_vals[0][0]

            # For d_i
            # Max d_i after removing idx
            if idx != max_d_vals[0][1]:
                max_d = max_d_vals[0][0]
            else:
                max_d = max_d_vals[1][0] if max_d_vals[1][1] != -1 else max_d_vals[0][0]

            # Min d_i after removing idx
            if idx != min_d_vals[0][1]:
                min_d = min_d_vals[0][0]
            else:
                min_d = min_d_vals[1][0] if min_d_vals[1][1] != -1 else min_d_vals[0][0]

            dist = max(max_s - min_s, max_d - min_d)
            min_dist = min(min_dist, dist)

        return min_dist