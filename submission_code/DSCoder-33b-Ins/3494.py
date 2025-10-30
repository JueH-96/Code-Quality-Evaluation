class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort()
        verticalCut.sort()
        horizontalCut.append(m)
        verticalCut.append(n)
        h_max = max(horizontalCut[0], m - horizontalCut[0])
        v_max = max(verticalCut[0], n - verticalCut[0])
        for i in range(1, len(horizontalCut)):
            h_max = max(h_max, horizontalCut[i] - horizontalCut[i-1])
        for i in range(1, len(verticalCut)):
            v_max = max(v_max, verticalCut[i] - verticalCut[i-1])
        return (h_max * v_max) % (10**9 + 7)