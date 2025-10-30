class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        
        def min_cost_to_make_uniform(s):
            discrepancies = []
            for p in range(n - 1):
                if s[p] != s[p + 1]:
                    discrepancies.append(p)
            
            if not discrepancies:
                return 0
            
            discrepancies.sort()
            total_cost = 0
            covered_up_to = -1
            i = 0
            while i < len(discrepancies):
                p_j = discrepancies[i]
                if p_j > covered_up_to:
                    cost_prefix = p_j + 2  # prefix flip up to p_j + 1
                    cost_suffix = n - p_j - 1  # suffix flip from p_j + 1
                    min_cost = min(cost_prefix, cost_suffix)
                    total_cost += min_cost
                    if min_cost == cost_prefix:
                        covered_up_to = p_j
                    else:
                        covered_up_to = n - 2
                else:
                    i += 1
                    continue
                # Remove all discrepancies up to covered_up_to
                while i < len(discrepancies) and discrepancies[i] <= covered_up_to:
                    i += 1
            return total_cost
        
        # Cost to make all '0's
        cost_zeros = min_cost_to_make_uniform(s)
        # Invert the string to make all '1's
        inverted_s = ''.join('1' if c == '0' else '0' for c in s)
        cost_ones = min_cost_to_make_uniform(inverted_s)
        
        return min(cost_zeros, cost_ones)