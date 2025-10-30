from typing import List

class Solution:
  def numberOfSequence(self, n: int, sick: List[int]) -> int:
    MOD = 10**9 + 7

    # Constraints:
    # 2 <= n <= 10^5
    # 1 <= sick.length <= n - 1
    # This implies N_healthy = n - sick.length is always >= 1.
    # The case n == sick.length (all are sick, N_healthy = 0) is not possible under these constraints.

    # Precompute factorials and inverse factorials.
    # Max N_healthy can be n-1. Max individual segment length L can also be n-1.
    # So, factorials up to n-1 are needed.
    # Using n as max_val_for_fact means arrays of size n+1, indices 0 to n.
    # This covers up to fact[n] and invfact[n].
    max_val_for_fact = n 
    fact = [1] * (max_val_for_fact + 1)
    invfact = [1] * (max_val_for_fact + 1)

    for i in range(1, max_val_for_fact + 1):
        fact[i] = (fact[i-1] * i) % MOD
    
    # invfact[k] = (k!)^(-1) mod MOD
    invfact[max_val_for_fact] = pow(fact[max_val_for_fact], MOD - 2, MOD)
    # We use the property: invfact[k-1] = ((k-1)!)^(-1) = (k!/k)^(-1) = (k!)^(-1) * k = invfact[k] * k
    for i in range(max_val_for_fact, 0, -1): 
        invfact[i-1] = (invfact[i] * i) % MOD

    segment_lengths = []
    power_of_2_multiplier = 1

    # First segment: children from 0 to sick[0]-1
    # Length of this segment is sick[0].
    # This segment is infected from one side (by sick[0]). Its internal infection order is fixed.
    if sick[0] > 0: # If sick[0] is 0, this segment has length 0.
        segment_lengths.append(sick[0])
    
    # Middle segments: children between sick[i] and sick[i+1]
    # These are children from sick[i]+1 to sick[i+1]-1.
    # Length = (sick[i+1]-1) - (sick[i]+1) + 1 = sick[i+1] - sick[i] - 1.
    # These segments can be infected from two sides.
    for i in range(len(sick) - 1):
        length = sick[i+1] - sick[i] - 1
        if length > 0:
            segment_lengths.append(length)
            # Number of ways to infect its children internally is 2^(length-1).
            # If length is 1, 2^(1-1) = 2^0 = 1 way. pow(2,0,MOD) correctly gives 1.
            term = pow(2, length - 1, MOD)
            power_of_2_multiplier = (power_of_2_multiplier * term) % MOD
    
    # Last segment: children from sick[len(sick)-1]+1 to n-1
    # Length = (n-1) - (sick[len(sick)-1]+1) + 1 = n - 1 - sick[len(sick)-1].
    # This segment is infected from one side (by sick[len(sick)-1]). Its internal infection order is fixed.
    if sick[len(sick)-1] < n - 1: # If sick[len(sick)-1] is n-1, this segment has length 0.
        length = (n - 1) - sick[len(sick)-1]
        segment_lengths.append(length)

    # Total number of children to be infected
    N_healthy = n - len(sick)

    # Based on constraints, N_healthy is always at least 1.
    # If N_healthy could be 0 (e.g. if sick.length == n was allowed),
    # the result should be 1 (one way: the empty sequence of infections).
    # The code below would correctly yield 1 if N_healthy = 0:
    # fact[0]=1, segment_lengths would be empty, power_of_2_multiplier=1.
    # So, an explicit `if N_healthy == 0: return 1` is not strictly necessary
    # but might be added for clarity if constraints were different.

    # Calculate multinomial coefficient: N_healthy! / (l1! * l2! * ... * lk!)
    # This is equivalent to N_healthy! * invfact[l1] * invfact[l2] * ...
    # This counts ways to interleave the infection of these segments.
    
    multinomial_coeff = fact[N_healthy]
    for L_val in segment_lengths:
        # L_val must be > 0 because we only add positive lengths to segment_lengths.
        # So invfact[L_val] refers to a precomputed valid inverse factorial.
        multinomial_coeff = (multinomial_coeff * invfact[L_val]) % MOD
        
    total_ways = (multinomial_coeff * power_of_2_multiplier) % MOD
    
    return total_ways