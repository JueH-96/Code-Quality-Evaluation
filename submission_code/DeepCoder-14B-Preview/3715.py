import bisect

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        if not coins:
            return 0
        
        # Sort the coins by their starting position
        coins_sorted = sorted(coins, key=lambda x: x[0])
        n = len(coins_sorted)
        
        # Generate the list of x's to consider, including l_i, r_i, and r_i + 1
        x_set = set()
        for seg in coins_sorted:
            x_set.add(seg[0])    # l_i
            x_set.add(seg[1])    # r_i
            x_set.add(seg[1] + 1)  # r_i + 1
        x_list = sorted(x_set)
        
        max_sum = 0
        
        # Precompute the list of l's for binary search
        l_list = [seg[0] for seg in coins_sorted]
        
        # Check for windows that are entirely within a single segment
        for seg in coins_sorted:
            l, r, c = seg
            if r - l + 1 >= k:
                current = c * k
                if current > max_sum:
                    max_sum = current
        
        # Process each x in x_list
        for x in x_list:
            right = x + k - 1
            # Find the rightmost segment where l <= right
            idx = bisect.bisect_right(l_list, right) - 1
            current_sum = 0
            
            # Iterate through all segments up to the found index
            for i in range(idx + 1):
                l, r, c = coins_sorted[i]
                if r < x:
                    continue
                start = max(l, x)
                end = min(r, right)
                if start > end:
                    continue
                current_sum += (end - start + 1) * c
            
            if current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum