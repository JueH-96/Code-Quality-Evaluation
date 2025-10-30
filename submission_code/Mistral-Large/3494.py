from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts to minimize the cost
        horizontalCut.sort()
        verticalCut.sort()

        # Helper function to calculate the minimum cost for a given set of cuts
        def calculate_cost(cuts):
            cost = 0
            prev_cut = 0
            for cut in cuts:
                cost += cut
                if cut > prev_cut:
                    cost += prev_cut
                prev_cut = cut
            return cost

        # Calculate the total cost for horizontal and vertical cuts
        horizontal_cost = calculate_cost(horizontalCut)
        vertical_cost = calculate_cost(verticalCut)

        # The total cost is the sum of horizontal and vertical costs
        return horizontal_cost + vertical_cost