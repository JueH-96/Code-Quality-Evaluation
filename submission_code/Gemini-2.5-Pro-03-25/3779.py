import math
from typing import List

class Solution:
  """
  Solves the pizza eating problem to maximize total weight gained.
  """
  def maxWeight(self, pizzas: List[int]) -> int:
    """
    Finds the maximum total weight gained by eating all pizzas optimally.

    Every day, 4 pizzas are eaten. Let the weights be W, X, Y, Z such that W <= X <= Y <= Z.
    On odd-numbered days (1, 3, ...), the weight gained is Z (the maximum).
    On even-numbered days (2, 4, ...), the weight gained is Y (the second maximum).
    The goal is to maximize the total weight gained after eating all pizzas.

    Args:
        pizzas: A list of integers representing the weight of the pizzas.
                It's guaranteed that len(pizzas) is a multiple of 4, and len(pizzas) >= 4.

    Returns:
        The maximum total weight gained.
    """

    n = len(pizzas)
    # Sort the pizzas by weight in ascending order. This allows us to use a greedy approach
    # based on the largest and smallest pizzas.
    pizzas.sort()

    # Calculate the total number of days (k). Since we eat 4 pizzas a day and n is a multiple of 4,
    # k is n / 4.
    k = n // 4

    # Calculate the number of odd-numbered days (1st, 3rd, 5th, ...).
    # The number of odd days is ceil(k/2).
    num_odd_days = (k + 1) // 2
    # The number of even days is floor(k/2). num_even_days = k // 2

    # Initialize the total weight gained.
    total_gain = 0

    # Strategy:
    # We need to form k groups of 4 pizzas [W, X, Y, Z] where W <= X <= Y <= Z.
    # The gain depends on the day number: Z for odd days, Y for even days.
    # The total gain is Sum(Z_i for odd days i) + Sum(Y_i for even days i).

    # Consider the 2k largest pizzas: pizzas[n-2k:]. These pizzas must constitute
    # the Y and Z values for the k groups formed over k days.
    # Why? If a Y value came from pizzas[:n-2k], say pizzas[a], and a W or X value
    # came from pizzas[n-2k:], say pizzas[b], then pizzas[a] <= pizzas[b].
    # But in any group [W, X, Y, Z], W <= X <= Y <= Z. This suggests the smallest
    # 2k elements form W and X, and the largest 2k elements form Y and Z.

    # Let's focus on the largest 2k elements: `large = pizzas[n-2k:]`.
    # We need to partition `large` into k pairs (Y_j, Z_j) such that Y_j <= Z_j.
    # A crucial insight is the optimal pairing strategy: Pair the smallest element
    # of `large` with the largest, the second smallest with the second largest, etc.
    # Pair j (for j from 0 to k-1) is formed by:
    # Y_j = pizzas[n - 2*k + j]  ( (j+1)-th smallest element in `large` )
    # Z_j = pizzas[n - 1 - j]    ( (j+1)-th largest element in `large` )
    # This pairing uses each element in `large` exactly once. We can verify Y_j <= Z_j:
    # n - 2*k + j <= n - 1 - j  =>  2j <= 2k - 1 => j <= k - 0.5. This holds for j = 0..k-1.

    # Now we have k conceptual pairs (Y_j, Z_j). We need to assign these k pairs to the k days.
    # Assigning pair j to an odd day yields gain Z_j.
    # Assigning pair j to an even day yields gain Y_j.
    # To maximize the total gain, we should assign the pairs that give the largest "boost"
    # from being on an odd day. The boost for pair j is (Z_j - Y_j).
    # We should assign the `num_odd_days` pairs with the largest (Z_j - Y_j) difference
    # to the odd days.

    # Calculation approach:
    # 1. Calculate the base gain assuming all k pairs were assigned to even days: Sum(Y_j for all j).
    # 2. Calculate the potential extra gain for each pair if assigned to an odd day: diff_j = Z_j - Y_j.
    # 3. Sort these differences in descending order.
    # 4. Add the top `num_odd_days` differences to the base gain.

    # List to store the differences (Z_j - Y_j).
    diffs = []

    # Calculate the base gain (Sum of all Y_j) and the differences.
    for j in range(k):
        # Y value for pair j
        y_j = pizzas[n - 2 * k + j]
        # Z value for pair j
        z_j = pizzas[n - 1 - j]

        # Add the base gain (Y_j contribution)
        total_gain += y_j
        # Calculate and store the difference (potential extra gain for odd days)
        diffs.append(z_j - y_j)

    # Sort the differences in descending order. The pairs corresponding to the largest
    # differences provide the most benefit when assigned to odd days.
    diffs.sort(reverse=True)

    # Add the extra gain from assigning the 'num_odd_days' pairs with the largest
    # differences to the odd-numbered days.
    for i in range(num_odd_days):
        # Add the i-th largest difference to the total gain.
        # Since k >= 1 (because n>=4) and num_odd_days <= k, diffs[i] is always a valid index.
        total_gain += diffs[i]

    # The final total_gain correctly represents Sum(Z_j for odd days) + Sum(Y_j for even days).
    return total_gain