import math

class Solution:
  """
  Finds the maximum value of (a XOR x) * (b XOR x) modulo 10^9 + 7,
  where 0 <= x < 2^n.
  """
  def maximumXorProduct(self, a: int, b: int, n: int) -> int:
    """
    Calculates the maximum XOR product using a greedy bitwise approach.

    The core idea is to determine the bits of A = (a XOR x) and B = (b XOR x) from the 
    most significant bit downwards to maximize their product A * B.
    The choice of the i-th bit of x (x_i) affects the i-th bits of A and B (A_i, B_i).
    For bits i >= n, x_i must be 0, so A_i and B_i are fixed by a_i and b_i.
    For bits i < n, we can choose x_i = 0 or x_i = 1. We make a greedy choice at each step
    to maximize the final product.

    Args:
      a: First integer.
      b: Second integer.
      n: The number of bits constraint for x (0 <= x < 2^n).

    Returns:
      The maximum product modulo 10^9 + 7.
    """

    MOD = 10**9 + 7

    # Initialize the potential values for A = a XOR x and B = b XOR x
    # We build these values bit by bit from MSB to LSB.
    # `current_A` and `current_B` store the values constructed so far using bits > i.
    current_A = 0
    current_B = 0

    # Iterate through bits from 49 down to 0. The maximum value for a, b is < 2^50,
    # so 50 bits (indices 0 to 49) are sufficient. n is also <= 50.
    # Iterating down ensures we make greedy choices based on the most significant bits first.
    for i in range(49, -1, -1):
        # Get the i-th bit of a and b
        a_i = (a >> i) & 1
        b_i = (b >> i) & 1

        # If the current bit index is greater than or equal to n,
        # the corresponding bit in x must be 0 (since 0 <= x < 2^n).
        if i >= n:
            # x_i = 0 is fixed.
            # The resulting bits for A and B are determined solely by a and b.
            # A_i = a_i XOR 0 = a_i
            # B_i = b_i XOR 0 = b_i
            # Set the i-th bit of current_A and current_B according to a_i and b_i.
            # The |= operation adds 2^i if the bit is 1, and adds 0 if the bit is 0.
            current_A |= (a_i << i)
            current_B |= (b_i << i)
        else:
            # If the current bit index is less than n, we have a choice for x_i (0 or 1).
            # A_i = a_i XOR x_i
            # B_i = b_i XOR x_i
            
            if a_i == b_i:
                # If a_i and b_i are the same, then A_i will equal B_i regardless of x_i.
                # A_i = a_i XOR x_i
                # B_i = b_i XOR x_i = a_i XOR x_i
                # To maximize the product A * B, we want A and B to be as large as possible.
                # Setting A_i = B_i = 1 contributes the maximum possible value `(1 << i)` to both A and B.
                # This is always achievable by choosing x_i appropriately:
                # If a_i=0, choose x_i=1 => A_i = 0 XOR 1 = 1, B_i = 0 XOR 1 = 1.
                # If a_i=1, choose x_i=0 => A_i = 1 XOR 0 = 1, B_i = 1 XOR 0 = 1.
                # Thus, we greedily set the i-th bit of both current_A and current_B to 1.
                current_A |= (1 << i)
                current_B |= (1 << i)
            else: # a_i != b_i
                # If a_i and b_i differ, then A_i != B_i regardless of x_i.
                # A_i XOR B_i = (a_i XOR x_i) XOR (b_i XOR x_i) = a_i XOR b_i = 1.
                # One of A_i, B_i must be 1, and the other must be 0.
                # We apply a greedy strategy: assign the '1' bit to the number
                # (A or B) that is currently smaller based on the higher bits processed so far.
                # Rationale: To maximize P = A*B, it's generally better if A and B are closer in value.
                # Giving the '1' bit (which represents a value of 2^i) to the currently smaller number
                # helps balance them.
                # Let A' = current_A, B' = current_B. We compare potential products after this bit:
                # Option (A_i=1, B_i=0): A becomes A' | (1 << i), B remains B'. Product approx (A' + 2^i) * B'.
                # Option (A_i=0, B_i=1): A remains A', B becomes B' | (1 << i). Product approx A' * (B' + 2^i).
                # Comparing the increments: (A' + 2^i) * B' vs A' * (B' + 2^i) is equivalent to comparing B' vs A'.
                # If current_A > current_B, choose option (A_i=0, B_i=1) which corresponds to larger increment 2^i*A'.
                # If current_A <= current_B, choose option (A_i=1, B_i=0) which corresponds to larger increment 2^i*B'.
                
                if current_A > current_B:
                    # current_A is larger. Assign the '1' bit to B to potentially make B larger. Set B_i = 1, A_i = 0.
                    # Update current_B by setting its i-th bit. current_A's i-th bit remains 0.
                    current_B |= (1 << i)
                else: # current_A <= current_B
                    # current_B is larger or equal. Assign the '1' bit to A. Set A_i = 1, B_i = 0.
                    # Update current_A by setting its i-th bit. current_B's i-th bit remains 0.
                    current_A |= (1 << i)

    # After iterating through all bits, current_A and current_B hold the values
    # of (a XOR x) and (b XOR x) that yield the maximum product according to the greedy strategy.
    # Calculate the final product modulo MOD. Python handles large integers automatically.
    # The modulo operation is applied at the end as required.
    # Using (current_A % MOD) * (current_B % MOD) % MOD is mathematically equivalent to
    # (current_A * current_B) % MOD for non-negative integers.
    result = (current_A % MOD) * (current_B % MOD) % MOD
    return result