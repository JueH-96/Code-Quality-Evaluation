class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[1])
        n_offers = len(offers)
        dp = [0] * n_offers
        for i in range(n_offers):
            current_offer = offers[i]
            gold_i = current_offer[2]
            start_i = current_offer[0]
            
            compatible_offer_index = -1
            low = 0
            high = i - 1
            while low <= high:
                mid = (low + high) // 2
                if offers[mid][1] < start_i:
                    compatible_offer_index = mid
                    low = mid + 1
                else:
                    high = mid - 1
                    
            prev_gold = 0
            if compatible_offer_index != -1:
                prev_gold = dp[compatible_offer_index]
                
            option2_gold = gold_i + prev_gold
            option1_gold = dp[i-1] if i > 0 else 0
            
            dp[i] = max(option1_gold, option2_gold)
            
        return dp[n_offers - 1] if n_offers > 0 else 0