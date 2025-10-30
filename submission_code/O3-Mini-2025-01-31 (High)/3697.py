from math import gcd
from itertools import combinations, product
from typing import List

def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        # Let m be the number of target elements.
        m = len(target)
        full_mask = (1 << m) - 1  # we want to cover all targets; each target will be one bit
        
        # Precompute lcm_for_mask[b] for every non‐empty bitmask b representing a subset of target.
        # For b with bits set, we want LCM(target[j] for each j with (b >> j) & 1).
        lcm_for_mask = [0] * (full_mask + 1)
        for mask in range(1, full_mask + 1):
            cur = 1
            bit = 0
            tmp = mask
            while tmp:
                if tmp & 1:
                    cur = lcm(cur, target[bit])
                bit += 1
                tmp //= 2
            lcm_for_mask[mask] = cur
        
        # For each candidate “option”, we imagine taking one element from nums and incrementing it 
        # until it becomes a multiple of LCM(subset). For a chosen subset (represented by a bitmask),
        # an element x requires cost = ((x + L - 1) // L) * L - x, where L = LCM(the chosen targets).
        # Notice that a single transformed number can “cover” several targets if it is a multiple of LCM.
        #
        # We now want to choose a set of elements (each used only once) and assign each one a subset mask 
        # so that the union (bitwise OR) of those masks equals full_mask. Our goal is to minimize the total cost.
        #
        # Because the number of targets (m) is at most 4 there are at most 2^4-1=15 different non‐empty masks.
        # For each mask, we will keep only a few (up to 4) candidates (lowest cost and their index)
        # so that later, when we combine candidates from different masks, we can try out the best few.
        candidate = {mask: [] for mask in range(1, full_mask + 1)}
        max_candidates = 4  # since m <= 4, we only need to keep up to 4 best candidates per mask
        
        # Prepare a list of (mask, LCM) tuples so we don’t have to index repeatedly.
        masks_info = [(mask, lcm_for_mask[mask]) for mask in range(1, full_mask + 1)]
        
        # Process each element in nums. For each element and for each candidate mask,
        # compute the cost to bump it up to the next multiple of L = lcm_for_mask[mask].
        for i, x in enumerate(nums):
            for mask, L_val in masks_info:
                # Compute cost = (ceil(x / L_val) * L_val - x)
                k = (x + L_val - 1) // L_val
                cost_val = k * L_val - x
                cand_list = candidate[mask]
                # (It’s not expected that the same element i appears twice for a given mask,
                #  but we check anyway.)
                duplicate = False
                for c, idx in cand_list:
                    if idx == i:
                        duplicate = True
                        if cost_val < c:
                            # Replace with lower cost (and re-sort later)
                            cand_list.remove((c, idx))
                            cand_list.append((cost_val, i))
                            cand_list.sort(key=lambda t: t[0])
                        break
                if duplicate:
                    continue

                if len(cand_list) < max_candidates:
                    cand_list.append((cost_val, i))
                    cand_list.sort(key=lambda t: t[0])
                else:
                    # If this cost is lower than the worst stored candidate, add it and trim the list.
                    if cost_val < cand_list[-1][0]:
                        cand_list.append((cost_val, i))
                        cand_list.sort(key=lambda t: t[0])
                        while len(cand_list) > max_candidates:
                            cand_list.pop()
        
        # Now, we want to assign a candidate option for some selection of distinct masks so that
        # the union of the masks equals full_mask.
        # Because each candidate option “slot” is tied to a mask type, and the order of selection does not matter,
        # we enumerate all combinations (of 1 up to m candidate mask keys) of distinct mask types among the 15 possible,
        # check if their OR covers all targets and then try out the product of candidates from them.
        best = float('inf')
        candidate_keys = list(candidate.keys())  # these are the 15 possible masks (from 1 to full_mask)
        
        for r in range(1, m + 1):
            for masks_combo in combinations(candidate_keys, r):
                combo_or = 0
                for b in masks_combo:
                    combo_or |= b
                if combo_or != full_mask:
                    continue
                # For each mask in the combo, we have a list candidate[mask] of candidate options.
                # We'll try every combination of candidate selections (one per mask) and check that the indices
                # (i.e. the original nums elements) are distinct.
                groups = [candidate[b] for b in masks_combo]
                for selection in product(*groups):
                    # selection is a tuple of (cost, index) for each mask in masks_combo.
                    indices = [idx for (_, idx) in selection]
                    if len(set(indices)) < r:
                        continue
                    total_cost = sum(cost for (cost, _) in selection)
                    if total_cost < best:
                        best = total_cost
        return best