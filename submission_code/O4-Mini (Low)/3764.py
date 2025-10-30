from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        # Collect the top `limits[i]` elements from each row
        candidates = []
        for row, lim in zip(grid, limits):
            if lim <= 0:
                continue
            # sort row descending and take up to lim
            # if lim >= len(row), just take the whole row
            top_elems = sorted(row, reverse=True)[:lim]
            candidates.extend(top_elems)
        
        # Now we have a flat list of all allowed candidates;
        # take the k largest among them
        if not candidates or k <= 0:
            return 0
        
        # sort descending and sum the first k
        candidates.sort(reverse=True)
        return sum(candidates[:k])


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxSum([[1,2],[3,4]], [1,2], 2))  # 7
    print(sol.maxSum([[5,3,7],[8,2,6]], [2,2], 3))  # 21