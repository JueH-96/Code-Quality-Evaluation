from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 7:
            return 0  # As per constraints, but the problem says n >=7
        
        product_map = defaultdict(list)
        
        # Preprocess all valid (p, r) pairs and store in product_map
        for p in range(n):
            for r in range(p + 2, n):
                product = nums[p] * nums[r]
                product_map[product].append((r, p))
        
        # Process each product in product_map to group by r and sort p's
        for product in product_map:
            # Sort the list of (r, p) pairs by r
            product_map[product].sort()
            grouped = []
            i = 0
            while i < len(product_map[product]):
                r_current = product_map[product][i][0]
                p_list = []
                # Collect all p's with the same r
                while i < len(product_map[product]) and product_map[product][i][0] == r_current:
                    p_list.append(product_map[product][i][1])
                    i += 1
                p_list.sort()
                grouped.append((r_current, p_list))
            product_map[product] = grouped
        
        total = 0
        
        # Iterate over all valid (q, s) pairs where s >= q + 4
        for q in range(n):
            for s in range(q + 4, n):
                P = nums[q] * nums[s]
                if P not in product_map:
                    continue
                grouped_list = product_map[P]
                if not grouped_list:
                    continue
                
                # Extract r values from grouped_list
                r_vals = [entry[0] for entry in grouped_list]
                # Find indices in grouped_list where r >= q+2 and r <= s-2
                left = bisect_left(r_vals, q + 2)
                right = bisect_right(r_vals, s - 2)
                
                # Iterate through valid r entries and count valid p's
                for i in range(left, right):
                    r_val, p_list = grouped_list[i]
                    # Count p's <= q-2 using bisect
                    count = bisect_right(p_list, q - 2)
                    total += count
        
        return total