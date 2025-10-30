from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort both cost lists in descending order so that
        # the most expensive cuts are performed first.
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)

        h_i = v_i = 0                  # indices for horizontal / vertical cuts
        horizontal_pieces = 1          # current number of horizontal pieces (rows)
        vertical_pieces   = 1          # current number of vertical   pieces (columns)
        total_cost = 0

        # Continue until all cuts are performed
        while h_i < len(horizontalCut) or v_i < len(verticalCut):

            # Decide which cut to make next.
            # Pick the larger remaining cost
            take_horizontal = (
                v_i == len(verticalCut) or
                (h_i < len(horizontalCut) and horizontalCut[h_i] >= verticalCut[v_i])
            )

            if take_horizontal:
                # Cutting horizontally – cost is multiplied by current number of columns
                total_cost += horizontalCut[h_i] * vertical_pieces
                horizontal_pieces += 1
                h_i += 1
            else:
                # Cutting vertically – cost is multiplied by current number of rows
                total_cost += verticalCut[v_i] * horizontal_pieces
                vertical_pieces += 1
                v_i += 1

        return total_cost