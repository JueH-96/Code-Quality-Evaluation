from typing import List
from functools import cmp_to_key

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        """
        The problem is equivalent to scheduling n jobs on a single machine.
        ---------------------------------------------------------------
        • job   i  :   needs   t_i = ceil(health[i] / power)   seconds (hits)
        • weight i :   w_i  = damage[i]   (cost per second while the job is unfinished)
        
        Bob’s total damage is  Σ w_i * C_i  where  C_i  is the completion time
        (moment when enemy i is killed).  
        Minimising Σ w_i * C_i on a single machine with pre-emption allowed is
        solved by Smith’s Rule: order jobs by increasing  t_i / w_i.
        ---------------------------------------------------------------
        After sorting, we kill enemies one by one (no need to interleave),
        accumulate the running time and add w_i * current_time to the answer.
        """
        
        # Build (processing_time, weight) pairs
        pairs = []
        for d, h in zip(damage, health):
            shots = (h + power - 1) // power        # ceil division
            pairs.append((shots, d))                # (p_i, w_i)
        
        # Sort by ratio  p_i / w_i  without using floating point
        def cmp(a, b):
            p1, w1 = a
            p2, w2 = b
            diff = p1 * w2 - p2 * w1               # compare p1/w1  vs  p2/w2
            if diff < 0:   return -1
            if diff > 0:   return 1
            # tie-breaker (doesn’t influence optimality)
            return p1 - p2
        
        pairs.sort(key=cmp_to_key(cmp))
        
        # Simulate killing enemies in that order
        elapsed  = 0
        answer   = 0
        for p, w in pairs:
            elapsed += p
            answer  += w * elapsed
        
        return answer