class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Create a set of all possible amounts that can be made using the coins
        amounts = set()
        for coin in coins:
            amount = coin
            while amount <= 2 * 10**9:
                amounts.add(amount)
                amount += coin
        
        # Sort the amounts and return the kth smallest
        amounts = sorted(amounts)
        return amounts[k-1]