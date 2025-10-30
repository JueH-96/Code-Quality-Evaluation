class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts
        horizontalCut.sort()
        verticalCut.sort()

        # Calculate the maximum horizontal and vertical cuts
        max_horizontal = max(horizontalCut[0], m - horizontalCut[-1])
        max_vertical = max(verticalCut[0], n - verticalCut[-1])

        # Calculate the costs of the maximum horizontal and vertical cuts
        for i in range(len(horizontalCut) - 1):
            max_horizontal = max(max_horizontal, horizontalCut[i+1] - horizontalCut[i])
        for i in range(len(verticalCut) - 1):
            max_vertical = max(max_vertical, verticalCut[i+1] - verticalCut[i])

        # The minimum total cost is the product of the maximum horizontal and vertical cuts
        return max_horizontal * max_vertical