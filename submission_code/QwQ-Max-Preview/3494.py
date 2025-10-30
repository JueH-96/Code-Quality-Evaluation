class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the horizontal and vertical cuts in descending order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        h_ptr = 0
        v_ptr = 0
        h_count = 0  # Number of horizontal cuts made so far
        v_count = 0  # Number of vertical cuts made so far
        total_cost = 0
        
        while h_ptr < len(horizontalCut) or v_ptr < len(verticalCut):
            # Choose the next highest cut between horizontal and vertical
            if h_ptr < len(horizontalCut) and (v_ptr >= len(verticalCut) or horizontalCut[h_ptr] >= verticalCut[v_ptr]):
                total_cost += horizontalCut[h_ptr] * (v_count + 1)
                h_ptr += 1
                h_count += 1
            else:
                total_cost += verticalCut[v_ptr] * (h_count + 1)
                v_ptr += 1
                v_count += 1
        
        return total_cost