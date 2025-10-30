import math

class Solution:
  """
  This class provides a solution to find the minimum possible value 
  of the last element of a strictly increasing array of positive integers 
  of size n, such that the bitwise AND of all elements equals x.
  """
  def minEnd(self, n: int, x: int) -> int:
    """
    Constructs an array nums of size n such that nums consists of positive integers, 
    nums[i+1] > nums[i] for all 0 <= i < n-1, and the bitwise AND of all elements is x.
    Returns the minimum possible value of nums[n-1].

    Args:
        n: The size of the array.
        x: The target bitwise AND value.

    Returns:
        The minimum possible value for the last element nums[n-1].

    Problem Analysis:
    Let the array be nums = [a_0, a_1, ..., a_{n-1}].
    Conditions:
    1. a_i > 0 for all i.
    2. a_0 < a_1 < ... < a_{n-1}.
    3. a_0 & a_1 & ... & a_{n-1} = x.

    From condition 3, the bitwise AND of all elements is x. This implies that for any bit position `p`, if the `p`-th bit of `x` is 1, then the `p`-th bit of every `a_i` must be 1.
    This can be stated as `a_i & x == x` for all `i`.
    This condition also implies `a_i >= x`. Since `x >= 1` (from constraints), this ensures `a_i > 0`.

    Let S be the set of positive integers `a` such that `a & x == x`.
    The problem requires us to find a sequence `a_0, ..., a_{n-1}` such that `a_i \in S` for all `i`, `a_0 < a_1 < ... < a_{n-1}`, and `a_0 & ... & a_{n-1} = x`. We want to minimize `a_{n-1}`.

    Consider the elements of S sorted in increasing order: `s_0, s_1, s_2, ...`.
    The smallest element `s_0` must be `x` itself, because `x & x == x` and any `a` such that `a & x == x` must have `a >= x`.
    The sequence `nums = [s_0, s_1, ..., s_{n-1}]` satisfies the conditions:
    - `s_i > 0`: Since `x >= 1`, all `s_i >= x >= 1$.
    - `s_0 < s_1 < ... < s_{n-1}`: By definition of the sorted sequence.
    - `s_0 & s_1 & ... & s_{n-1} = x`: Since `s_0 = x`, and `s_i & x == x` for all `i`, the bitwise AND must be `x`. Any bit set in `x` is set in all `s_i`, so it's set in the result. Any bit not set in `x` is not set in `s_0=x`, so it's not set in the result.

    To minimize `a_{n-1}`, we must choose the smallest `n` distinct elements from `S`. These are exactly `s_0, s_1, ..., s_{n-1}`.
    Therefore, the minimum possible value for `a_{n-1}` is `s_{n-1}`.

    How to find `s_{n-1}`?
    The elements `a` in `S` are of the form `x | y`, where `y` has bits set only at positions where `x` has 0 bits.
    This means `a = x + V`, where `V` is a number whose binary representation has 1s only at positions where `x` has 0 bits (i.e., `V & x == 0`).
    Let the bit positions where `x` has 0 bits be `p_0 < p_1 < p_2 < ...`.
    The values `V` that can be formed are sums of distinct powers of 2 of the form `2^{p_j}`.
    Let the sequence of these `V` values sorted in increasing order be `V_0, V_1, V_2, ...`. Then `s_k = x + V_k`.
    The value `V_k` corresponds to the integer `k` where its bits are mapped to positions `p_j`.
    Specifically, if `k = sum(k_j * 2^j)` in binary, then `V_k = sum(k_j * 2^{p_j})`.

    We need to find `s_{n-1} = x + V_{n-1}`. Let `k = n - 1`. We compute `V_k`.

    Algorithm:
    1. Initialize `k = n - 1`. This is the 0-based index of the element we seek in the sorted sequence S.
    2. Initialize `V = 0`. This variable will accumulate the value `V_k`.
    3. Initialize `current_p = 0`. This tracks the current bit position being examined in `x`.
    4. Iterate while `k > 0`:
        a. Find the next available bit position `p >= current_p` where `x` has a 0 bit. Increment `current_p` until `(x >> current_p) & 1 == 0`.
        b. Check the least significant bit (LSB) of `k`. If `k & 1 == 1`:
            Set the `current_p`-th bit in `V` using `V |= (1 << current_p)`.
        c. Increment `current_p` to move to the next bit position for the next iteration's search.
        d. Right shift `k` by 1 (`k >>= 1`) to process its next bit.
    5. The final result is `x | V`. This combines the fixed bits from `x` with the bits determined by `k` into the available slots.
    """

    # Convert n to 0-based index k for finding the k-th element after the first (s_0 = x).
    k = n - 1
    
    # V will store the value V_k, which represents the bits added to x
    # based on the bits of k mapped to the available positions (where x has 0 bits).
    V = 0  
    
    # current_p tracks the current bit position we are examining in x.
    current_p = 0 
    
    # Iterate while there are bits remaining in k to process.
    # This loop processes the bits of k from least significant to most significant.
    while k > 0:
        # Find the next available bit position `p` (where p >= current_p) such that the p-th bit of x is 0.
        # We skip positions where x has a 1 bit because these positions are fixed by x.
        while (x >> current_p) & 1:
            current_p += 1
        
        # Check the least significant bit of the current value of k.
        if k & 1:
            # If the LSB of k is 1, it means we need to set the bit at the current available position `current_p` in our result offset V.
            # We use bitwise OR to set the bit, effectively adding 2^current_p to V.
            V |= (1 << current_p)
        
        # Move to the next bit position in x. This ensures that the next iteration
        # finds the next available 0-bit position for the next bit of k.
        current_p += 1
        
        # Move to the next bit of k by right shifting. This discards the LSB we just processed.
        k >>= 1
            
    # The final result is the n-th smallest number `s_{n-1}` in the sequence S.
    # This is obtained by combining the base value `x` with the calculated offset `V`.
    # Since V has bits set only where x has 0 bits, x & V = 0. Thus x | V = x + V.
    # We use bitwise OR as it clearly represents combining the bit patterns.
    return x | V