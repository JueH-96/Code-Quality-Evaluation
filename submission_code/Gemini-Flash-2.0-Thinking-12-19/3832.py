from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        """
        Calculates the minimum total cost to reach each position i (0 to n-1)
        starting from position n.

        Args:
            cost: An integer array of size n, where cost[i] is the cost to
                  swap with the person initially at position i.

        Returns:
            An array of size n, where answer[i] is the minimum total cost
            to reach position i.
        """
        n = len(cost)
        
        # answer[i] will store the minimum total cost to reach position i.
        # We are calculating this for positions 0, 1, ..., n-1.
        answer = [0] * n
        
        # Base case: To reach position 0 from position n (our starting point).
        # The most direct way involving a paid forward move is to swap with the
        # person initially at position 0. We are at position n, they are initially
        # at position 0. Since 0 < n, they are in front of us. Swapping with them costs cost[0].
        # After this swap, we are at position 0. This is the minimum cost for position 0.
        answer[0] = cost[0]
        
        # min_cost_to_reach_prev tracks the minimum cost found so far to reach any
        # position j, where j is strictly less than the current index i we are calculating for.
        # When calculating answer[i], this variable holds min(answer[0], answer[1], ..., answer[i-1]).
        # This value is crucial because if we can reach *any* position j < i
        # with a certain cost, we can then reach position i from j for free
        # by repeatedly swapping with people currently behind us (including the person
        # currently at position i, since i > j).
        min_cost_to_reach_prev = answer[0]
        
        # Iterate from position 1 up to n-1 to calculate the minimum cost for each.
        for i in range(1, n):
            # To find the minimum cost to reach position i, we consider two main strategies:
            #
            # Strategy 1: Make a paid swap specifically targeting position i.
            # The most straightforward paid swap to advance towards i from behind
            # is to swap with the person initially at position i. If this person
            # is still at their initial position i (and we are behind them,
            # having not yet reached or passed i), swapping costs cost[i].
            # This puts us directly at position i. The total cost for this path
            # starting from n could be cost[i].
            #
            # Strategy 2: Leverage a cheaper path to an earlier position.
            # If we have already reached *any* position j < i with a certain
            # minimum cost (tracked by min_cost_to_reach_prev), we can then
            # move from position j to position i for free. This is because
            # from position j, the person currently at position i (since i > j)
            # is behind us. Swapping with someone behind us costs 0.
            # Thus, the cost via this strategy is the minimum cost to reach any j < i,
            # which is min_cost_to_reach_prev.
            #
            # The minimum cost to reach position i is the minimum of the costs
            # resulting from these two strategies.
            answer[i] = min(cost[i], min_cost_to_reach_prev)
            
            # Update the minimum cost found so far for any position from 0 up
            # to and including the current index i. This updated minimum will be
            # used in the next iteration (i+1) as the minimum cost to reach any
            # position before i+1.
            # The new minimum is the minimum of the previous minimum (which covered 0..i-1)
            # and the newly calculated minimum cost for position i.
            min_cost_to_reach_prev = min(min_cost_to_reach_prev, answer[i])
            
        return answer