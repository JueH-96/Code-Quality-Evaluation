from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            # left tile index (wrap-around)
            left = (i - 1) % n
            # right tile index (wrap-around)
            right = (i + 1) % n
            # Check if current tile's color is different from both its neighbors.
            if colors[i] != colors[left] and colors[i] != colors[right]:
                count += 1
        return count

# Example test cases
if __name__ == "__main__":
    sol = Solution()
    # Test case 1: Expected output 0
    print(sol.numberOfAlternatingGroups([1, 1, 1]))
    # Test case 2: Expected output 3
    print(sol.numberOfAlternatingGroups([0, 1, 0, 0, 1]))