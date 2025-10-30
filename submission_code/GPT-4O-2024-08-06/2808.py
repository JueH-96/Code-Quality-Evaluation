class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        
        # Sort walls by cost/time ratio in descending order
        walls = sorted(range(n), key=lambda i: cost[i] / time[i], reverse=True)
        
        total_cost = 0
        paid_painter_time = 0
        
        for i in walls:
            if paid_painter_time <= 0:
                # Use the paid painter
                total_cost += cost[i]
                paid_painter_time = time[i]
            else:
                # Use the free painter
                paid_painter_time -= 1
        
        return total_cost