class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cut costs in decreasing order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        horizontal_segments = 1  # Initially we have 1 horizontal segment
        vertical_segments = 1    # Initially we have 1 vertical segment
        total_cost = 0
        
        h_index = 0
        v_index = 0
        
        # Make cuts until all are made
        while h_index < len(horizontalCut) or v_index < len(verticalCut):
            # If there are no more horizontal cuts or if the vertical cut is more expensive
            if h_index >= len(horizontalCut) or (v_index < len(verticalCut) and verticalCut[v_index] > horizontalCut[h_index]):
                # Make a vertical cut - cost is multiplied by current horizontal segments
                total_cost += verticalCut[v_index] * horizontal_segments
                vertical_segments += 1
                v_index += 1
            else:
                # Make a horizontal cut - cost is multiplied by current vertical segments
                total_cost += horizontalCut[h_index] * vertical_segments
                horizontal_segments += 1
                h_index += 1
        
        return total_cost