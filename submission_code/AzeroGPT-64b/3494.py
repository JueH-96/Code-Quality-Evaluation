class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        h = sorted([0] + horizontalCut + [m])
        v = sorted([0] + verticalCut + [n])
        hh = vv = 0
        for i in range(1, len(h)):
            hh = max(hh, h[i]-h[i-1])
        for i in range(1, len(v)):
            vv = max(vv, v[i]-v[i-1])
        return int(hh*vv % (1e9+7))