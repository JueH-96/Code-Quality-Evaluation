import math

class Solution:
  def waysToReachStair(self, k: int) -> int:
    """
    Calculates the total number of ways Alice can reach stair k.

    The method is based on a combinatorial analysis of the allowed moves.
    Let u be the number of 'up' moves and d be the number of 'down' moves.
    - Alice starts at stair 1.
    - u 'up' moves contribute a total displacement of (2^u - 1).
    - d 'down' moves contribute a total displacement of -d.
    The final stair k is given by: k = 1 + (2^u - 1) - d, which simplifies to k = 2^u - d.
    From this, for a given u, the required number of down moves is d = 2^u - k.

    The constraints on the moves are:
    1. Down moves cannot be consecutive.
    2. A down move is only allowed from a stair i > 0.

    The first constraint implies that the number of down moves 'd' cannot be
    more than one greater than the number of up moves 'u' (i.e., d <= u + 1).
    If this condition holds, the number of ways to arrange u 'U's and d 'D's
    without consecutive 'D's is given by the binomial coefficient C(u + 1, d).

    The second constraint is naturally satisfied by any sequence that meets
    the first constraint, as a down move from stair 0 would require a preceding
    down move, which is not allowed.

    The algorithm iterates through possible values of u, calculates the
    corresponding d, and if the conditions (d >= 0 and d <= u + 1) are met,
    it adds C(u + 1, d) to the total count.
    """
    total_ways = 0
    
    # We iterate through the number of up-moves 'u'.
    # The loop for u can be bounded because 2^u grows exponentially.
    # For k <= 10^9, u will not exceed ~35.
    for u in range(35):
        # The stair position after 'u' up-moves and 0 down-moves would be 2^u.
        position_after_ups = 1 << u
        
        # For a given u, the required number of down-moves is d = 2^u - k.
        # d must be non-negative, so 2^u must be at least k.
        if position_after_ups < k:
            continue
            
        d = position_after_ups - k
        
        # The number of down-moves 'd' cannot be more than u + 1.
        # If d > u + 1, C(u+1, d) is 0. Also, for all subsequent u, this
        # condition will also fail, so we can stop.
        if d > u + 1:
            break

        # The number of ways to arrange 'u' U's and 'd' D's with no consecutive D's
        # is C(u + 1, d).
        total_ways += math.comb(u + 1, d)
            
    return total_ways