class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: [int], verticalCut: [int]) -> int:
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        h, v, hcuts, vcuts = 0, 0, 1, 1
        cost = 0
        while h < len(horizontalCut) and v < len(verticalCut):
            if horizontalCut[h] > verticalCut[v]:
                cost += horizontalCut[h] * vcuts
                hcuts += 1
                h += 1
            else:
                cost += verticalCut[v] * hcuts
                vcuts += 1
                v += 1
        while h < len(horizontalCut):
            cost += horizontalCut[h] * vcuts
            h += 1
        while v < len(verticalCut):
            cost += verticalCut[v] * hcuts
            v += 1
        return cost