from typing import List
import math

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        # Initialize the answer array with infinity.
        # answer[i] will store the minimum cost to reach position i.
        answer = [math.inf] * n

        # Iterate through each possible initial paid swap.
        # We consider making the first paid swap with Person k (0 <= k < n).
        # Initially, You are at position n, and Person j is at position j for 0 <= j < n.
        # Swapping You@n with Pk@k costs cost[k].
        # After this initial swap: You are at position k, Pk is at position n,
        # and all other persons Pj (j != k) remain at their original positions j.
        
        for k in range(n):
            initial_paid_cost = cost[k]

            # Case 1: Reaching a target position i where i >= k.
            # From our current position k, any person Pj (where j > k) is currently
            # at their original position j, and thus is behind us (since j > k).
            # We can swap with any person behind us for free.
            # To reach position i (where k <= i < n), we can perform a sequence
            # of free swaps with the people currently at positions k+1, k+2, ..., i.
            # The total cost to reach any position i >= k via this initial swap with Pk
            # is just the initial paid cost.
            for i in range(k, n):
                answer[i] = min(answer[i], initial_paid_cost)

            # Case 2: Reaching a target position i where i < k.
            # We are at position k (cost = initial_paid_cost).
            # The people Pj (where j < k) are still at their original positions j,
            # which are in front of us (since j < k).
            # To move from position k towards position i (i < k), we must swap
            # with people in front of us. Swapping You@p with Pj@j (where j < p) costs cost[j].
            # The minimum-cost way to move from k down to i is by swapping sequentially
            # with the person currently at k-1, then k-2, ..., down to i.
            # After the initial swap with Pk, Person j (j < k) is still at position j.
            # The path is You@k -> swap P(k-1)@k-1 (cost cost[k-1]) -> You@k-1 ->
            # swap P(k-2)@k-2 (cost cost[k-2]) -> You@k-2 -> ... ->
            # swap Pi@i (cost cost[i]) -> You@i.
            # The total cost for these backward steps is cost[k-1] + cost[k-2] + ... + cost[i].
            # The total cost to reach position i < k via this initial swap with Pk is
            # initial_paid_cost + (cost[k-1] + cost[k-2] + ... + cost[i]).

            current_backward_sum = 0
            # Iterate from k-1 down to 0 (the target positions i < k)
            for i in range(k - 1, -1, -1):
                # When considering target position i, the costs for the backward steps
                # from k down to i are cost[k-1], cost[k-2], ..., cost[i].
                # As the loop index `i` decreases from k-1 down to 0,
                # `current_backward_sum` accumulates the costs cost[k-1] + ... + cost[i].
                # At index `i`, we add cost[i] to the sum that represents cost[k-1] + ... + cost[i+1].
                current_backward_sum += cost[i] 
                
                # The total cost via this path is the initial_paid_cost plus the accumulated backward sum.
                answer[i] = min(answer[i], initial_paid_cost + current_backward_sum)

        return answer