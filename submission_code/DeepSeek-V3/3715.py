class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort the coins based on the starting position
        coins.sort()
        
        # Initialize the prefix sum array
        prefix = []
        total = 0
        for l, r, c in coins:
            total += (r - l + 1) * c
            prefix.append(total)
        
        # Initialize the maximum coins
        max_coins = 0
        
        # Iterate through each segment to find the maximum sum of k consecutive bags
        for i in range(len(coins)):
            l_i, r_i, c_i = coins[i]
            # Determine the starting and ending positions for the window
            # The window starts at l_i and ends at l_i + k - 1
            window_start = l_i
            window_end = window_start + k - 1
            
            # Find the segments that overlap with the window
            # We need to find all segments that have any overlap with [window_start, window_end]
            # Since the segments are non-overlapping and sorted, we can use binary search
            # to find the first segment that ends after or at window_start
            # and the last segment that starts before or at window_end
            
            # Find the first segment that ends after or at window_start
            low = 0
            high = len(coins) - 1
            first = i
            while low <= high:
                mid = (low + high) // 2
                if coins[mid][1] >= window_start:
                    first = mid
                    high = mid - 1
                else:
                    low = mid + 1
            
            # Find the last segment that starts before or at window_end
            low = 0
            high = len(coins) - 1
            last = i
            while low <= high:
                mid = (low + high) // 2
                if coins[mid][0] <= window_end:
                    last = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            # Calculate the sum of coins in the window
            current_sum = 0
            for j in range(first, last + 1):
                l_j, r_j, c_j = coins[j]
                # Determine the overlap between [l_j, r_j] and [window_start, window_end]
                overlap_start = max(l_j, window_start)
                overlap_end = min(r_j, window_end)
                if overlap_start <= overlap_end:
                    current_sum += (overlap_end - overlap_start + 1) * c_j
            
            # Update the maximum coins
            if current_sum > max_coins:
                max_coins = current_sum
        
        return max_coins