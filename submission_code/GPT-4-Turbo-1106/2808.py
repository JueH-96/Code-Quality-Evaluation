class Solution:
    def paintWalls(self, cost: list[int], time: list[int]) -> int:
        # Pair the costs and times, and sort them in descending order of time
        # because we want to assign the longest tasks to the paid painter
        # when the free painter is available.
        paired = sorted(zip(cost, time), key=lambda x: -x[1])
        
        total_cost = 0
        free_painter_time = 0  # Time for which the free painter is occupied
        
        for c, t in paired:
            if free_painter_time < t:
                # If the free painter is available before the paid painter finishes,
                # assign the current wall to the paid painter.
                total_cost += c
                free_painter_time += 1  # Free painter will take 1 unit of time
            
        return total_cost

# Example usage:
# sol = Solution()
# print(sol.paintWalls([1,2,3,2], [1,2,3,2]))  # Output: 3
# print(sol.paintWalls([2,3,4,2], [1,1,1,1]))  # Output: 4