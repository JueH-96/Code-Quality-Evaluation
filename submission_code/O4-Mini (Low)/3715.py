from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort the coin segments by starting position
        coins.sort(key=lambda x: x[0])
        # Build a list of (length, weight) segments including zero-weight gaps
        arr = []
        # Add the first coin segment
        prev_l, prev_r, prev_c = coins[0]
        arr.append([prev_r - prev_l + 1, prev_c])
        # Now add gaps and subsequent coin segments
        for l, r, c in coins[1:]:
            gap = l - prev_r - 1
            if gap > 0:
                arr.append([gap, 0])
            arr.append([r - l + 1, c])
            prev_r = r
        
        # Two-pointer sliding window on arr to maintain a window of total length k
        L = 0             # left index in arr
        offset = 0        # how many units of arr[L] have been consumed
        curr_len = 0      # total length of current window
        curr_sum = 0      # total coin sum of current window
        ans = 0
        
        for R in range(len(arr)):
            length_R, w_R = arr[R]
            # Add the entire R-segment
            curr_len += length_R
            curr_sum += length_R * w_R
            
            # If window exceeds k in length, shrink from the left
            while curr_len > k:
                extra = curr_len - k
                length_L, w_L = arr[L]
                avail = length_L - offset
                if extra >= avail:
                    # remove the rest of arr[L]
                    curr_sum -= avail * w_L
                    curr_len -= avail
                    L += 1
                    offset = 0
                else:
                    # remove part of arr[L]
                    curr_sum -= extra * w_L
                    curr_len -= extra
                    offset += extra
            
            # If window exactly k, update answer
            if curr_len == k:
                ans = max(ans, curr_sum)
        
        return ans