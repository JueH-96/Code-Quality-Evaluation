class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the horizontal and vertical cuts in descending order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)

        # Keep track of how many horizontal and vertical pieces we have
        # Initially we have 1 horizontal segment and 1 vertical segment
        horizontal_pieces = 1
        vertical_pieces = 1
        total_cost = 0

        # Pointers to traverse horizontal and vertical cuts
        i, j = 0, 0
        while i < len(horizontalCut) and j < len(verticalCut):
            # Pick the larger cut cost from either horizontal or vertical
            if horizontalCut[i] >= verticalCut[j]:
                # Cutting horizontally costs horizontalCut[i] * (number of vertical pieces)
                total_cost += horizontalCut[i] * vertical_pieces
                horizontal_pieces += 1
                i += 1
            else:
                # Cutting vertically costs verticalCut[j] * (number of horizontal pieces)
                total_cost += verticalCut[j] * horizontal_pieces
                vertical_pieces += 1
                j += 1

        # If there are leftover horizontal cuts
        while i < len(horizontalCut):
            total_cost += horizontalCut[i] * vertical_pieces
            horizontal_pieces += 1
            i += 1

        # If there are leftover vertical cuts
        while j < len(verticalCut):
            total_cost += verticalCut[j] * horizontal_pieces
            vertical_pieces += 1
            j += 1

        return total_cost