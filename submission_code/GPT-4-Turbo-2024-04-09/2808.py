class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        # Combine the cost and time into a list of tuples and sort by time descending
        walls = sorted(zip(cost, time), key=lambda x: -x[1])
        
        total_cost = 0
        paid_time = 0
        free_time = 0
        
        for c, t in walls:
            if paid_time <= free_time:
                # If the paid painter is free or both are equally busy, use the paid painter
                total_cost += c
                paid_time += t
            else:
                # Otherwise, use the free painter
                free_time += 1
        
        return total_cost