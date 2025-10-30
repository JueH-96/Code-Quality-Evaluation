class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers based on the ending house index
        # Each offer is of the form [start, end, gold]
        offers.sort(key=lambda x: x[1])
        
        # Prepare a list of the ending indices (for binary search)
        ends = [offer[1] for offer in offers]
        
        # dp[i] will represent the maximum gold we can get
        # considering up to the i-th offer (in sorted order by end).
        dp = [0] * (len(offers) + 1)
        
        # For binary searching the "j" that does not overlap:
        # we want the largest j < i such that offers[j].end < offers[i].start
        import bisect
        
        for i in range(1, len(offers) + 1):
            start_i, end_i, gold_i = offers[i - 1]
            
            # Find the rightmost offer j that ends < start_i
            # We do this with bisect on the ends array:
            # We want the index j s.t. ends[j] < start_i
            # bisect_left finds the leftmost index to insert start_i
            # so ends[j] < start_i for j < that insertion point
            j = bisect.bisect_left(ends, start_i)  # j is the count of offers ending < start_i
            
            # Option 1: Skip the i-th offer
            # Option 2: Take the i-th offer, add dp[j] to gold_i
            dp[i] = max(dp[i-1], dp[j] + gold_i)
        
        return dp[len(offers)]