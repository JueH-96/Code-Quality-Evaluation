import sys
from itertools import combinations

class Solution:
    def makeStringGood(self, s: str) -> int:
        counts = [0] * 26
        for c in s:
            idx = ord(c) - ord('a')
            counts[idx] += 1
        
        total_length = len(s)
        min_ops = float('inf')
        
        for k in range(1, 27):
            # Generate all possible combinations of k distinct characters
            for S in combinations(range(26), k):
                # Calculate sum_in and sum_out
                sum_in = sum(counts[i] for i in S)
                sum_out = total_length - sum_in
                
                # Determine a reasonable range for m
                if k == 0:
                    continue
                min_m = 1
                max_m = (sum_in + sum_out) // k + 100  # Adding buffer to cover possible m's
                
                for m in range(min_m, max_m + 1):
                    sum_deficit = 0
                    sum_surplus = 0
                    for i in S:
                        if counts[i] < m:
                            sum_deficit += (m - counts[i])
                        else:
                            sum_surplus += (counts[i] - m)
                    
                    base_cost = sum_out + sum_deficit + sum_surplus
                    
                    # Calculate cost reduction by changing characters to next in the subset
                    cost_reduction = 0
                    # Create a set for faster look-up
                    S_set = set(S)
                    for i in S:
                        next_i = i + 1
                        if next_i < 26 and next_i in S_set:
                            surplus = max(0, counts[i] - m)
                            deficit = max(0, m - counts[next_i])
                            possible = min(surplus, deficit)
                            cost_reduction += possible * 1  # Each change saves 1 operation
                    
                    total_cost = base_cost - cost_reduction
                    if total_cost < min_ops:
                        min_ops = total_cost
        
        return min_ops if min_ops != float('inf') else 0