class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        paid_painter_cost = 0
        free_painter_cost = 0
        paid_painter_time = 0
        
        # Sort the walls based on time taken
        walls = sorted(zip(time, cost), key=lambda x: x[0])
        
        # Assign walls to paid painter
        for t, c in walls:
            if paid_painter_time + t <= t:
                paid_painter_cost += c
                paid_painter_time += t
            else:
                free_painter_cost += c
        
        return paid_painter_cost + free_painter_cost