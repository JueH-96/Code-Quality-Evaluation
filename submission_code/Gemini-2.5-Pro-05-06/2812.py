class Solution:
  def theMaximumAchievableX(self, num: int, t: int) -> int:
    """
    Calculates the maximum possible achievable number x.

    An integer x is achievable if it can become equal to num after applying
    the following operation no more than t times:
    Increase or decrease x by 1, and simultaneously increase or decrease num by 1.

    Args:
      num: The initial value of num.
      t: The maximum number of operations allowed. Can be 0.

    Returns:
      The maximum possible achievable number x.
    """

    # Based on the derivation above, the maximum achievable x is num + 2*t.
    # Let x_achievable be the candidate for the maximum achievable number.
    # Let num_initial be the given initial value of num.
    # After k operations (k <= t), where x changes by delta_x_i and num by delta_num_i:
    #   x_final = x_achievable + sum(delta_x_i)
    #   num_final = num_initial + sum(delta_num_i)
    # If x_final = num_final, then:
    #   x_achievable = num_initial + sum(delta_num_i - delta_x_i)
    # To maximize x_achievable, we maximize sum(delta_num_i - delta_x_i).
    # Each term (delta_num_i - delta_x_i) is maximized when delta_num_i = +1 and delta_x_i = -1.
    # In this case, (delta_num_i - delta_x_i) = 2.
    # We perform this for t operations (the maximum allowed).
    # So, max sum = 2 * t.
    # Therefore, max_x_achievable = num_initial + 2 * t.

    return num + 2 * t