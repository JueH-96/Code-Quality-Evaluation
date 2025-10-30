class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        import sys

        # Compute projections
        proj1 = [x + y for x, y in points]
        proj2 = [x - y for x, y in points]
        proj3 = [-x + y for x, y in points]
        proj4 = [-x - y for x, y in points]

        # Function to find stats for a projection list
        def find_proj_stats(proj):
            max_val = float('-inf')
            min_val = float('inf')
            count_max = 0
            count_min = 0
            for val in proj:
                if val > max_val:
                    max_val = val
                    count_max = 1
                elif val == max_val:
                    count_max += 1
                if val < min_val:
                    min_val = val
                    count_min = 1
                elif val == min_val:
                    count_min += 1
            second_max = float('-inf')
            for val in proj:
                if val < max_val:
                    if val > second_max:
                        second_max = val
            second_min = float('inf')
            for val in proj:
                if val > min_val:
                    if val < second_min:
                        second_min = val
            return max_val, min_val, second_max, second_min, count_max, count_min

        # Get stats for each projection
        proj1_stats = find_proj_stats(proj1)
        proj2_stats = find_proj_stats(proj2)
        proj3_stats = find_proj_stats(proj3)
        proj4_stats = find_proj_stats(proj4)

        # Function to get new max and min after removing a point's value
        def get_new_max_min(val, max_val, min_val, second_max, second_min, count_max, count_min):
            new_max = max_val if val != max_val or count_max > 1 else second_max
            new_min = min_val if val != min_val or count_min > 1 else second_min
            return new_max, new_min

        # Compute new maximum distances after removing each point
        new_max_distances = []
        for i in range(len(points)):
            p1 = proj1[i]
            p2 = proj2[i]
            p3 = proj3[i]
            p4 = proj4[i]

            new_max_p1, new_min_p1 = get_new_max_min(p1, *proj1_stats)
            new_max_p2, new_min_p2 = get_new_max_min(p2, *proj2_stats)
            new_max_p3, new_min_p3 = get_new_max_min(p3, *proj3_stats)
            new_max_p4, new_min_p4 = get_new_max_min(p4, *proj4_stats)

            new_dist_p1 = new_max_p1 - new_min_p1
            new_dist_p2 = new_max_p2 - new_min_p2
            new_dist_p3 = new_max_p3 - new_min_p3
            new_dist_p4 = new_max_p4 - new_min_p4

            new_max_distance = max(new_dist_p1, new_dist_p2, new_dist_p3, new_dist_p4)
            new_max_distances.append(new_max_distance)

        # Find and return the minimum of the new maximum distances
        result = min(new_max_distances)
        return result