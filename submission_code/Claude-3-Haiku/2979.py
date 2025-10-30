class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers by the end_i in ascending order
        offers.sort(key=lambda x: x[1])
        
        # Initialize the maximum profit
        max_profit = 0
        
        # Iterate through the offers
        for start, end, gold in offers:
            # Check if the current offer can be accepted
            if all(start <= offer[1] for offer in offers if offer[0] <= end):
                max_profit += gold
                
                # Mark the houses in the current offer as sold
                for i in range(start, end + 1):
                    offers = [offer for offer in offers if offer[0] > end or offer[1] < start]
        
        return max_profit