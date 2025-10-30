class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)

        h_segments = 1
        v_segments = 1
        h_index = 0
        v_index = 0
        cost = 0

        while h_index < len(horizontalCut) or v_index < len(verticalCut):
            if h_index < len(horizontalCut) and (v_index == len(verticalCut) or horizontalCut[h_index] > verticalCut[v_index]):
                cost += horizontalCut[h_index] * v_segments
                h_segments += 1
                h_index += 1
            else:
                cost += verticalCut[v_index] * h_segments
                v_segments += 1
                v_index += 1

        return cost