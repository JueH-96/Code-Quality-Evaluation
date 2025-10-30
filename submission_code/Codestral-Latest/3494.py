class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cut arrays
        horizontalCut.sort()
        verticalCut.sort()

        # Calculate the maximum horizontal and vertical distances
        max_horizontal = max(horizontalCut[0], m - horizontalCut[-1])
        for i in range(1, len(horizontalCut)):
            max_horizontal = max(max_horizontal, horizontalCut[i] - horizontalCut[i - 1])
        max_vertical = max(verticalCut[0], n - verticalCut[-1])
        for i in range(1, len(verticalCut)):
            max_vertical = max(max_vertical, verticalCut[i] - verticalCut[i - 1])

        # The result is the product of the maximum distances modulo 10^9 + 7
        return (max_horizontal * max_vertical) % (10**9 + 7)