from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        x_plus_y = []
        x_minus_y = []
        for p in points:
            x_plus_y.append(p[0] + p[1])
            x_minus_y.append(p[0] - p[1])

        # Function to find top two max and min for a list
        def get_extremes(arr):
            max1 = max2 = -float('inf')
            min1 = min2 = float('inf')
            count_max1 = count_min1 = 0
            for num in arr:
                if num > max1:
                    max2 = max1
                    max1 = num
                    count_max1 = 1
                elif num == max1:
                    count_max1 += 1
                elif num > max2:
                    max2 = num

                if num < min1:
                    min2 = min1
                    min1 = num
                    count_min1 = 1
                elif num == min1:
                    count_min1 += 1
                elif num < min2:
                    min2 = num
            return max1, count_max1, max2, min1, count_min1, min2

        # Get extremes for x_plus_y and x_minus_y
        max1_p, count_max1_p, max2_p, min1_p, count_min1_p, min2_p = get_extremes(x_plus_y)
        max1_m, count_max1_m, max2_m, min1_m, count_min1_m, min2_m = get_extremes(x_minus_y)

        result = float('inf')
        for i in range(n):
            # For x_plus_y
            if x_plus_y[i] == max1_p:
                if count_max1_p == 1:
                    new_max_p = max2_p
                else:
                    new_max_p = max1_p
            else:
                new_max_p = max1_p

            if x_plus_y[i] == min1_p:
                if count_min1_p == 1:
                    new_min_p = min2_p
                else:
                    new_min_p = min1_p
            else:
                new_min_p = min1_p

            # For x_minus_y
            if x_minus_y[i] == max1_m:
                if count_max1_m == 1:
                    new_max_m = max2_m
                else:
                    new_max_m = max1_m
            else:
                new_max_m = max1_m

            if x_minus_y[i] == min1_m:
                if count_min1_m == 1:
                    new_min_m = min2_m
                else:
                    new_min_m = min1_m
            else:
                new_min_m = min1_m

            # Calculate the maximum distance after removing point i
            current_max_dist = max(new_max_p - new_min_p, new_max_m - new_min_m)
            result = min(result, current_max_dist)

        return result