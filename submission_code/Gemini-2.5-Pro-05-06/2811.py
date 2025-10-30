class Solution:
  def minimumSum(self, n: int, k: int) -> int:
    current_sum = 0
    # count of elements added to the k-avoiding array
    count = 0
    # Use a set to store elements already added to the k-avoiding array.
    # This allows for efficient O(1) average time complexity lookups.
    added_elements = set()
    
    # num_to_add will be the candidate integer to add to our array.
    # We start checking from 1, as we want positive integers and the smallest possible sum.
    num_to_add = 1
    
    # We continue until we have n elements in our k-avoiding array.
    while count < n:
      # The k-avoiding condition: "there does not exist any pair of distinct elements that sum to k".
      # If we add `num_to_add` to our array, then for any element `x` already in `added_elements`,
      # `num_to_add + x` must not be equal to `k`.
      # This means `num_to_add` must not be equal to `k - x` for any `x` in `added_elements`.
      # Rephrased: if `k - num_to_add` is an element `x` that is already in `added_elements`,
      # then `num_to_add` cannot be added.
      #
      # This check `(k - num_to_add) not in added_elements` correctly handles all cases:
      # 1. If `k - num_to_add` is a positive integer already in `added_elements`:
      #    `num_to_add` is skipped to prevent `(k - num_to_add) + num_to_add = k`.
      # 2. If `k - num_to_add == num_to_add` (i.e. `num_to_add = k/2`):
      #    The check becomes `num_to_add not in added_elements`. Since `num_to_add` is currently
      #    being considered and not yet added, this condition is true. So, `k/2` is added.
      #    This is correct because the pair `(k/2, k/2)` does not consist of distinct elements.
      # 3. If `k - num_to_add <= 0` (i.e. `num_to_add >= k`):
      #    `k - num_to_add` will not be in `added_elements` (which only stores positive integers).
      #    So `num_to_add` will be added. This is correct because if `num_to_add >= k`, and `x` is
      #    any positive integer in `added_elements`, then `num_to_add + x > k` (since `x >= 1`).
      #
      # Additional array properties (distinct positive integers) are maintained:
      # - Positive: `num_to_add` starts at 1 and increases.
      # - Distinct: `num_to_add` increases, so we only consider adding each number once.
      
      if (k - num_to_add) not in added_elements:
        added_elements.add(num_to_add)
        current_sum += num_to_add
        count += 1
      
      # Move to the next candidate integer.
      num_to_add += 1
            
    return current_sum