from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        """
        Calculates the maximum total weight gained from eating pizzas.
        n is the number of pizzas, guaranteed to be a multiple of 4.
        Each day, 4 pizzas are eaten. Group of 4 pizzas {W, X, Y, Z} (sorted W <= X <= Y <= Z).
        On odd-numbered days (1-indexed), the weight gained is Z.
        On even-numbered days (1-indexed), the weight gained is Y.

        To maximize the total weight gained over n/4 days, we should make
        the values contributed (Z on odd days, Y on even days) as large as possible.

        First, sort the pizzas in ascending order: p_0 <= p_1 <= ... <= p_{n-1}.

        Across all n/4 days, we use all n pizzas exactly once. Each day uses 4 pizzas.
        From any group of 4 pizzas {W, X, Y, Z}, the two smallest (W and X) never contribute
        to the gained weight. The contributing pizzas are Y and Z.

        This suggests that the smallest n/2 pizzas (p_0 to p_{n/2 - 1}) can be used
        for the W and X positions across the n/4 groups without losing any potential
        gain, as they would contribute less than the larger pizzas if placed in Y or Z positions.
        The largest n/2 pizzas (p_{n/2} to p_{n-1}) must then fill the Y and Z positions
        across the n/4 groups.

        Consider the n/2 largest pizzas: q_0, q_1, ..., q_{n/2-1}, where q_j = p_{n/2 + j}.
        These q pizzas are partitioned into n/4 pairs {y_i, z_i} such that y_i <= z_i,
        where i ranges from 0 to n/4 - 1. Each pair is assigned to day i+1.

        On day i+1 (1-indexed):
        If i+1 is odd (i is even): the gain is z_i.
        If i+1 is even (i is odd): the gain is y_i.

        The total weight gained is sum(z_i for i in {0, 2, 4, ...}) + sum(y_i for i in {1, 3, 5, ...}),
        where i iterates through the 0-indexed day numbers.

        Consider the pairing strategy {y_i, z_i} = {q_i, q_{n/2 - 1 - i}} for i = 0, ..., n/4 - 1.
        Since q is sorted, q_i <= q_{n/2 - 1 - i} for i < n/4.
        So, y_i = q_i and z_i = q_{n/2 - 1 - i}.

        This pairing strategy seems to maximize the sum. The sum includes:
        From odd days (i even): q[n/2 - 1], q[n/2 - 3], q[n/2 - 5], ...
        From even days (i odd): q[1], q[3], q[5], ...
        The indices of q being summed are {1, 3, 5, ...} union {n/2-1, n/2-3, ...}.
        This union simplifies to the set of odd indices in the range [0, n/2 - 1]: {1, 3, 5, ..., n/2 - 1}.

        The sum is sum(q[j] for j in range(1, n//2, 2)).
        Substitute q[j] = p_{n/2 + j}:
        The sum is sum(p[n//2 + j] for j in range(1, n//2, 2)).

        Let k = n//2 + j. When j goes 1, 3, 5, ..., k goes n//2+1, n//2+3, n//2+5, ...
        The loop for k starts at n//2 + 1.
        The last j is n//2 - 1 if n//2 is even, or n//2 - 2 if n//2 is odd.
        The corresponding last k is n//2 + (n//2 - 1) = n - 1 if n//2 is even,
        or n//2 + (n//2 - 2) = n - 2 if n//2 is odd.
        This covers exactly the indices n//2 + 1, n//2 + 3, ..., n - 1.

        Therefore, the maximum total weight is the sum of the pizza weights at indices
        n//2 + 1, n//2 + 3, ..., n - 1 in the sorted list of pizzas.
        """

        # Sort the pizzas in ascending order
        pizzas.sort()

        n = len(pizzas)
        total_weight = 0

        # Sum the elements at indices n/2 + 1, n/2 + 3, ..., n-1
        # These indices correspond to the pizzas that contribute to the sum
        # according to the optimal strategy derived above.
        # The range starts from n//2 + 1 (inclusive) and goes up to n (exclusive)
        # with a step of 2.
        # For example, if n=8, n//2=4, range(5, 8, 2) -> indices 5, 7.
        # If n=12, n//2=6, range(7, 12, 2) -> indices 7, 9, 11.
        # If n=4, n//2=2, range(3, 4, 2) -> index 3.
        for i in range(n // 2 + 1, n, 2):
            total_weight += pizzas[i]

        return total_weight