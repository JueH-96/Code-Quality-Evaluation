class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontal_cuts_sorted = sorted([(cost, 'h', i) for i, cost in enumerate(horizontalCut)], key=lambda x: x[0], reverse=True)
        vertical_cuts_sorted = sorted([(cost, 'v', j) for j, cost in enumerate(verticalCut)], key=lambda x: x[0], reverse=True)
        cuts = horizontal_cuts_sorted + vertical_cuts_sorted
        cuts.sort(key=lambda x: x[0], reverse=True)

        total_cost = 0
        horizontal_segments = 1
        vertical_segments = 1

        for cost, cut_type, _ in cuts:
            if cut_type == 'h':
                if horizontalCut[_] != -1: # check if this cut has been used already. Not necessary based on problem description, but good practice to avoid re-cutting.
                    total_cost += cost * vertical_segments
                    horizontal_segments += 1
                    horizontalCut[_] = -1 # mark as used, not strictly needed for this problem but good practice in general
            elif cut_type == 'v':
                if verticalCut[_] != -1: # check if this cut has been used already
                    total_cost += cost * horizontal_segments
                    vertical_segments += 1
                    verticalCut[_] = -1 # mark as used

        return total_cost