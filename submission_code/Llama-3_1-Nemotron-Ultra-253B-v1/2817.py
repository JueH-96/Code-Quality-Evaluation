class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        
        def calculate_cost(target):
            cost = 0
            i = 0
            flip = 0  # 0: even number of flips, 1: odd
            while i < n:
                current = s[i]
                if flip % 2 == 1:
                    current = '1' if current == '0' else '0'
                if current == str(target):
                    i += 1
                    continue
                # Need to flip. Find the best option between prefix and suffix
                # Calculate cost for flipping prefix up to j or suffix starting at i
                # Find the next position where the character matches after flip
                # We can flip prefix up to j (current i) or suffix starting at i
                cost_prefix = i + 1
                cost_suffix = n - i
                if cost_prefix < cost_suffix:
                    cost += cost_prefix
                    flip += 1
                    i += 1
                else:
                    cost += cost_suffix
                    i = n  # suffix flip covers the rest
            return cost
        
        cost0 = calculate_cost(0)
        cost1 = calculate_cost(1)
        return min(cost0, cost1)