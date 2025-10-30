from heapq import heappop, heappush
from typing import List

class FoodPair:
    def __init__(self, dish_idx, is_main, val):
        self.dish_idx = dish_idx
        self.is_main = is_main
        self.val = val

    def __lt__(self, other):
        return self.val < other.val

class Solution:
    def expensiveMeal(self, n: int, m: int, k: int, main_dishes: List[int], side_dishes: List[int], skip_pairs: List[List[int]]) -> int:
        main_max = [FoodPair(i, True, main_dishes[i]) for i in range(n)]
        side_max = [FoodPair(i, False, side_dishes[i]) for i in range(m)]
        for i in range(n):
            heappush(main_max, FoodPair(i, True, main_dishes[i]))
        for i in range(m):
            heappush(side_max, FoodPair(i, False, side_dishes[i]))
        main_2nd_max = []
        side_2nd_max = []
        skip_set = set((pair[0] - 1, pair[1] - 1) for pair in skip_pairs)
        while True:
            main_val = heappop(main_max)
            if not main_2nd_max:
                main_2nd_max.append(main_val)
            elif main_2nd_max[0].val < main_val.val:
                heappush(main_2nd_max, main_val)
                heappop(main_2nd_max)
            side_val = heappop(side_max)
            if not side_2nd_max:
                side_2nd_max.append(side_val)
            elif side_2nd_max[0].val < side_val.val:
                heappush(side_2nd_max, side_val)
                heappop(side_2nd_max)
            max_meal_cost = main_val.val + side_val.val
            if (main_val.dish_idx, side_val.dish_idx) not in skip_set and (main_2nd_max[0].dish_idx, side_2nd_max[0].dish_idx) not in skip_set:
                return max_meal_cost
            heappush(main_max, main_val)
            heappush(side_max, side_val)

# Example usage
solution = Solution()
print(solution.expensiveMeal(2, 3, 3, [2, 1], [10, 30, 20], [[1, 2], [2, 1], [2, 3]])) # Should print 31
print(solution.expensiveMeal(2, 1, 0, [1000000000, 1], [1000000000], [])) # Should print 2000000000
print(solution.expensiveMeal(10, 10, 10, 
                                [47718, 21994, 74148, 76721, 98917, 73766, 29598, 59035, 69293, 29127],
                                [7017, 46004, 16086, 62644, 74928, 57404, 32168, 45794, 19493, 71590],
                                [[1, 3], [2, 6], [4, 5], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 10], [7, 3]])) # Should print 149076