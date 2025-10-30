class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts in descending order based on their cost
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # Initialize total cost and the number of segments in each direction
        total_cost = 0
        horizontal_segments = 1
        vertical_segments = 1
        
        # Use a pointer for horizontal and vertical cuts
        h_idx = 0
        v_idx = 0
        
        # Process cuts in a greedy manner, always choosing the most expensive cut that affects the most pieces
        while h_idx < len(horizontalCut) and v_idx < len(verticalCut):
            if horizontalCut[h_idx] > verticalCut[v_idx]:
                # Cut horizontally
                total_cost += horizontalCut[h_idx] * vertical_segments
                horizontal_segments += 1
                h_idx += 1
            else:
                # Cut vertically
                total_cost += verticalCut[v_idx] * horizontal_segments
                vertical_segments += 1
                v_idx += 1
        
        # Process remaining horizontal cuts
        while h_idx < len(horizontalCut):
            total_cost += horizontalCut[h_idx] * vertical_segments
            horizontal_segments += 1
            h_idx += 1
        
        # Process remaining vertical cuts
        while v_idx < len(verticalCut):
            total_cost += verticalCut[v_idx] * horizontal_segments
            vertical_segments += 1
            v_idx += 1
        
        return total_cost