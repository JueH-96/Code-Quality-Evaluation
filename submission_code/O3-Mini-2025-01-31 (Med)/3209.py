from typing import List
import math

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # We want to “cover” all fruits 1 through n (1-indexed) by choosing a set of purchases.
        # When you purchase the fruit at index i (1-indexed) for prices[i-1] coins,
        # you get that fruit plus its next i fruits for free.
        # That is, a purchase at fruit i “covers” fruits i through min(n, i + i).
        #
        # We can view the problem as covering the interval [1, n] by a (possibly overlapping)
        # sequence of intervals. The twist is that you are allowed to buy a fruit even if you already
        # have it (i.e. it is already free) because its offer might extend your coverage further.
        # However, to keep things “contiguous” we must cover all fruits.
        #
        # Define dp[c] to be the minimum cost needed to guarantee that fruits 1..c are acquired.
        # Initially, dp[0] = 0 (nothing acquired) and dp[c] for c>=1 is set to infinity.
        #
        # Now, from a state where fruits 1..c are acquired, what moves do we have?
        # In a valid purchasing sequence the fruits you buy (their indices) must be non-decreasing.
        # Notice that if a fruit is not yet acquired, you can purchase it as long as it is exactly the next fruit,
        # i.e. fruit c+1. In addition, you are allowed to purchase any fruit that is already acquired (any index 1..c)
        # because its offer might extend further than your current coverage.
        #
        # When you purchase fruit i (an allowed move) the free offer covers fruits i through min(n, i+i).
        # We make the move only if it extends our coverage beyond c.
        
        INF = math.inf
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for c in range(n + 1):
            if dp[c] == INF:
                continue
            
            # Build the list of candidate fruit indices we can purchase.
            # For c == 0, no fruits are acquired yet so we must purchase fruit 1.
            # For c > 0, we may purchase any fruit in the already acquired block (1..c)
            # and we are also allowed to purchase fruit c+1 (the next uncovered fruit)
            candidates = []
            if c == 0:
                if 1 <= n:
                    candidates.append(1)
            else:
                for i in range(1, c + 1):
                    candidates.append(i)
                if c + 1 <= n:
                    candidates.append(c + 1)
            
            for fruit in candidates:
                # A purchase at fruit "fruit" covers from fruit to fruit+fruit.
                new_covered = fruit + fruit
                if new_covered > n:
                    new_covered = n
                # We only care if this purchase extends our current coverage.
                if new_covered > c:
                    dp[new_covered] = min(dp[new_covered], dp[c] + prices[fruit - 1])
        
        return dp[n]

# Example testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1: Expected output 4
    print(sol.minimumCoins([3, 1, 2]))
    # Example 2: Expected output 2
    print(sol.minimumCoins([1, 10, 1, 1]))