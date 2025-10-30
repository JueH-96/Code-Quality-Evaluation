from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # helper function to check possibility given a sort key and coordinate indices.
        # sort_key is the index of the “start” coordinate (0 for x, 1 for y),
        # and other indices: end_key = start_key+? For horizontal cuts,
        # start_y is index 1 and end_y is index 3; for vertical cuts,
        # start_x is index 0 and end_x is index 2.
        def can_cut(rectangles: List[List[int]], start_key: int, end_key: int) -> bool:
            m = len(rectangles)
            # sort by the "start" coordinate (start_x or start_y)
            recs = sorted(rectangles, key=lambda r: r[start_key])
            prefix_max = [0] * m
            prefix_max[0] = recs[0][end_key]
            for i in range(1, m):
                # maximum end coordinate seen so far
                prefix_max[i] = max(prefix_max[i-1], recs[i][end_key])
            # a valid "gap" (cut) exists between i and i+1 if:
            # the maximum ending coordinate among indices 0..i is strictly less than
            # the starting coordinate of rectangle at index i+1.
            splits = []
            for i in range(m-1):
                if prefix_max[i] < recs[i+1][start_key]:
                    splits.append(i)
            # We need two splits that partition the sorted list into three non-empty groups.
            # That is, we need to choose s1 and s2 from splits (with s1 < s2) such that:
            # group 1: indices 0..s1, group2: s1+1..s2, group3: s2+1..m-1 (each group non-empty)
            if len(splits) < 2:
                return False
            # Because splits is in sorted order (indices are increasing),
            # the first valid split s1 must be at least 0 and we need a second split s2 such that s2 < m-1.
            # Also, note that if the first split is at index s1, then group2 is non-empty only if s2 > s1.
            for i in range(len(splits)-1):
                s1 = splits[i]
                s2 = splits[i+1]
                if s1 >= 0 and s2 > s1 and s2 < m-1:
                    return True
            return False
        
        # Check horizontal candidate:
        horizontal_possible = can_cut(rectangles, start_key=1, end_key=3)
        # Check vertical candidate:
        vertical_possible = can_cut(rectangles, start_key=0, end_key=2)
        return horizontal_possible or vertical_possible

# Sample test runs:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    n = 5
    rectangles = [[1,0,5,2],[0,2,2,4],[3,2,5,3],[0,4,4,5]]
    print(sol.checkValidCuts(n, rectangles))  # Expected True

    # Example 2:
    n = 4
    rectangles = [[0,0,1,1],[2,0,3,4],[0,2,2,3],[3,0,4,3]]
    print(sol.checkValidCuts(n, rectangles))  # Expected True

    # Example 3:
    n = 4
    rectangles = [[0,2,2,4],[1,0,3,2],[2,2,3,4],[3,0,4,2],[3,2,4,4]]
    print(sol.checkValidCuts(n, rectangles))  # Expected False