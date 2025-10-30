from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        
        # answer[i] will store the minimum total cost to reach position i.
        answer = [0] * n
        
        # Base case: To reach position 0, we must make a paid swap with person 0 from our
        # starting position n. Person 0 is "in front" (0 < n).
        answer[0] = cost[0]
        
        # min_cost_so_far keeps track of the minimum cost to reach any position j where j < i.
        # If we can reach j for min_cost_so_far, and j < i, then person i (who is at i) 
        # is "behind" j (i > j). Thus, we can move from j to i for free.
        min_cost_so_far = answer[0]
        
        # Iterate from i = 1 to n-1
        for i in range(1, n):
            # Option 1: Make a direct paid swap from initial position n to i. Cost = cost[i].
            #           (Person i is "in front" of n, since i < n)
            # Option 2: Reach a position j (where j < i) with cost min_cost_so_far, 
            #           then swap to i for free. Cost = min_cost_so_far.
            #           (Person i is "behind" j, since i > j)
            answer[i] = min(cost[i], min_cost_so_far)
            
            # Update min_cost_so_far to include the newly calculated answer[i].
            # This ensures that for the next iteration (i+1), min_cost_so_far will correctly
            # represent the minimum of all answer[0]...answer[i].
            min_cost_so_far = min(min_cost_so_far, answer[i])
            
        return answer