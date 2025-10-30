class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        cuts = []
        for i, cost in enumerate(horizontalCut):
            cuts.append((cost, 'h', i))
        for i, cost in enumerate(verticalCut):
            cuts.append((cost, 'v', i))
        
        cuts.sort(key=lambda x: x[0], reverse=True)
        
        horizontal_pieces = 1
        vertical_pieces = 1
        total_cost = 0
        
        for cost, direction, index in cuts:
            if direction == 'h':
                total_cost += cost * vertical_pieces
                horizontal_pieces += 1
            else:
                total_cost += cost * horizontal_pieces
                vertical_pieces += 1
                
        return total_cost