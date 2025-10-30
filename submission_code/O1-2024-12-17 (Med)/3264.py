class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        
        # We'll repeatedly "buy" points by doing the first operation on
        # the smallest-cost enemy, as many times as possible in one go.
        # If we can no longer "buy" (i.e., first-operation) because of low energy,
        # but we have at least 1 point, we "sell" (second-operation) the largest
        # unmarked enemy to boost our energy. This mimics a strategy of
        # maximizing points with the cheapest cost, and refilling energy
        # with the most expensive ones, until no more gains are possible.
        
        l, r = 0, len(enemyEnergies) - 1
        points = 0
        max_points = 0
        
        while l <= r:
            # First, try as many first-operations on the smallest cost enemy
            cmin = enemyEnergies[l]
            if cmin > currentEnergy:
                # Can't afford even the smallest cost enemy
                # If we have at least 1 point and there is a larger enemy left,
                # try second operation to gain energy. Otherwise, we're done.
                if points >= 1 and l < r:
                    currentEnergy += enemyEnergies[r]
                    r -= 1
                else:
                    break
            else:
                # We can afford the smallest cost. Do as many first-operations
                # as possible in a single batch.
                times = currentEnergy // cmin
                points += times
                currentEnergy -= times * cmin
                max_points = max(max_points, points)
        
        return max_points