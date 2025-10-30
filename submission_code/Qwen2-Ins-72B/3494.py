class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.extend([0, m])
        verticalCut.extend([0, n])
        horizontalCut.sort()
        verticalCut.sort()
        
        max_h = max(horizontalCut[i+1] - horizontalCut[i] for i in range(len(horizontalCut)-1))
        max_v = max(verticalCut[i+1] - verticalCut[i] for i in range(len(verticalCut)-1))
        
        return (max_h * max_v) % (10**9 + 7)