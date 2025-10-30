import bisect

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers based on their end positions
        offers.sort(key=lambda x: x[1])
        
        # Initialize the ends and max_gold lists with a dummy entry (-1, 0)
        ends = [-1]
        max_gold = [0]
        
        for s, e, g in offers:
            # Find the rightmost end that is <= s-1 using binary search
            target = s - 1
            idx = bisect.bisect_right(ends, target) - 1
            current_gold = max_gold[idx] + g
            
            # Update the ends and max_gold lists if the current_gold is higher than the last recorded maximum
            if current_gold > max_gold[-1]:
                ends.append(e)
                max_gold.append(current_gold)
        
        return max_gold[-1]