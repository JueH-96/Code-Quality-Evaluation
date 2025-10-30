from typing import List
import math

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        """
        Calculates the minimum cost to reach each position i (0 to n-1) starting from position n.

        Args:
            cost: A list of integers where cost[i] is the cost to swap with
                  the person originally at position i when they are in front.

        Returns:
            A list of integers where answer[i] is the minimum total cost
            to reach position i.
        """
        n = len(cost)
        # Constraints state 1 <= n <= 100, so cost is never empty.

        answer = [0] * n

        # min_so_far will track the minimum cost[j] for all j < i.
        # Initialize with infinity so that the first element cost[0] is correctly picked
        # as the minimum consideration for reaching position 0.
        min_so_far = float('inf')

        # Iterate through each position i from 0 to n-1 that we want to reach.
        for i in range(n):
            # To reach position i, we can consider two main strategies that result in minimum cost:
            #
            # 1. Perform a direct initial swap with Original_Person_i from our starting position n.
            #    Original_Person_i is originally at position i. Since i < n, Original_Person_i
            #    is in front of our starting position n. According to the rules, swapping with
            #    someone in front costs cost[i]. After this swap, we are at position i.
            #    The total cost for this path is cost[i].
            #
            # 2. Perform an initial swap with some Original_Person_j from our starting position n,
            #    where j is an index less than i (j < i).
            #    The cost for this initial swap is cost[j] (since Original_Person_j is at original
            #    position j < n, in front of n). After this swap, we are at position j, and Original_Person_j
            #    moves to position n. All other people k (including Original_Person_i, since i != j)
            #    remain at their original positions k.
            #    Now we are at position j, and Original_Person_i is still at their original position i.
            #    Since i > j, Original_Person_i is currently behind our position j. According to the rules,
            #    swapping with someone behind us costs 0.
            #    This second swap takes us from position j to position i.
            #    The total cost for this path is cost[j] (first swap) + 0 (second swap) = cost[j].
            #    To minimize the cost using this strategy, we should choose j < i such that cost[j] is minimal.
            #    The minimum cost using strategy 2 is min(cost[j] for j in range(i)).
            #    Our variable `min_so_far` stores exactly this value (the minimum of cost[0] through cost[i-1])
            #    when we are processing index i in the loop.
            #
            # A strategy involving an initial swap with Original_Person_j where j > i would cost
            # cost[j] to reach position j, and then cost[i] to reach position i from j (since i < j,
            # P_i is in front of us at position j). The total cost would be cost[j] + cost[i], which
            # is greater than or equal to cost[i] (since costs are positive), making it suboptimal
            # compared to the direct swap (strategy 1).

            # The minimum cost to reach position i is therefore the minimum of the costs from strategy 1 and strategy 2.
            # min(cost[i], min(cost[j] for j in range(i)))
            # This is calculated by min(cost[i], min_so_far), where min_so_far holds min(cost[0]...cost[i-1]).
            answer[i] = min(cost[i], min_so_far)

            # Update min_so_far for the next iteration (for position i+1).
            # For the next iteration, we need the minimum cost among cost[0] through cost[i].
            # This is the current min_so_far (min(cost[0]...cost[i-1])) minimum with cost[i].
            min_so_far = min(min_so_far, cost[i])

        return answer