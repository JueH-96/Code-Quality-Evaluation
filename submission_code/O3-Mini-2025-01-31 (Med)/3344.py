from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        # When we remove one point, the maximum Manhattan distance among the remaining points
        # equals max(new_A_range, new_B_range), where A = x+y and B = x-y.
        # For each projection, the extremes are determined by:
        #   A_min = min(points[i][0] + points[i][1]) and A_max = max(points[i][0] + points[i][1])
        #   B_min = min(points[i][0] - points[i][1]) and B_max = max(points[i][0] - points[i][1])
        #
        # Removing a point might “remove” an extreme value. Therefore, for each candidate removal,
        # we compute the new extreme values. Note that if a candidate is removed and it appears
        # at an extreme, then the next extreme in order will define the bound.
        #
        # Instead of checking every point (which is O(n)), we note that only points that are
        # extreme in A or B might affect the overall maximum distance. Therefore, we consider only
        # candidates that are among the two smallest/largest for A and B.
        
        # We'll build sorted lists for A and B values, each as list of (value, index)
        A = []
        B = []
        for i, (x, y) in enumerate(points):
            A.append((x + y, i))
            B.append((x - y, i))
        
        A.sort()  # sort by A value
        B.sort()  # sort by B value
        
        # Global extremes (without removals)
        A_min_val, A_min_idx = A[0]
        A_max_val, A_max_idx = A[-1]
        B_min_val, B_min_idx = B[0]
        B_max_val, B_max_idx = B[-1]
        global_distance = max(A_max_val - A_min_val, B_max_val - B_min_val)
        
        # Consider candidate indices only among extreme values: first two and last two from A, and same for B.
        candidate_set = set()
        for arr in (A, B):
            # first two: if available
            candidate_set.add(arr[0][1])
            if n >= 2:
                candidate_set.add(arr[1][1])
            # last two:
            candidate_set.add(arr[-1][1])
            if n >= 2:
                candidate_set.add(arr[-2][1])
        
        def get_new_extremes(sorted_list, candidate):
            # Returns (new_min, new_max) of a sorted list (value,index) after removal of candidate
            # Check if the candidate is the extreme or if there are duplicate values.
            # new_min:
            if sorted_list[0][1] == candidate:
                # If there is a duplicate at the beginning, then still the min remains the same.
                if n >= 2 and sorted_list[1][0] == sorted_list[0][0]:
                    new_min = sorted_list[0][0]
                else:
                    new_min = sorted_list[1][0]
            else:
                new_min = sorted_list[0][0]
            
            # new_max:
            if sorted_list[-1][1] == candidate:
                # If duplicate exists:
                if n >= 2 and sorted_list[-2][0] == sorted_list[-1][0]:
                    new_max = sorted_list[-1][0]
                else:
                    new_max = sorted_list[-2][0]
            else:
                new_max = sorted_list[-1][0]
            
            return new_min, new_max
        
        best = float('inf')
        # Only test candidate removals that might improve the distance.
        for candidate in candidate_set:
            new_A_min, new_A_max = get_new_extremes(A, candidate)
            new_B_min, new_B_max = get_new_extremes(B, candidate)
            candidate_metric = max(new_A_max - new_A_min, new_B_max - new_B_min)
            best = min(best, candidate_metric)
        # It is possible that none of these candidates improve upon global_distance when none of them are extreme.
        # But we have to remove exactly one point, so best is our answer.
        return best


# You can use the following code to test the solution with the given examples:

if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    points1 = [[3,10],[5,15],[10,2],[4,4]]
    print(sol.minimumDistance(points1))  # Expected output: 12
    # Example 2:
    points2 = [[1,1],[1,1],[1,1]]
    print(sol.minimumDistance(points2))  # Expected output: 0