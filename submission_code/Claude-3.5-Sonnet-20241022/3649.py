class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        
        # Helper function to check if we can break all locks within given time
        def canBreakLocks(mask, time, factor, energy):
            # If all locks are broken (mask is 0), return True
            if mask == 0:
                return True
            
            # If we've seen this state before with same or better conditions
            state = (mask, time, factor, energy)
            if state in memo:
                return memo[state]
            
            # Try breaking any available lock
            for i in range(n):
                if mask & (1 << i):  # If lock i is not broken
                    new_energy = energy + factor
                    if new_energy >= strength[i]:
                        # Break this lock and continue with updated state
                        if canBreakLocks(mask ^ (1 << i), time + 1, factor + K, 0):
                            memo[state] = True
                            return True
            
            # If we can't break any lock, try accumulating more energy
            if canBreakLocks(mask, time + 1, factor, energy + factor):
                memo[state] = True
                return True
            
            memo[state] = False
            return False
        
        # Binary search for minimum time
        left, right = 1, sum(strength)  # Right bound can be optimized further
        memo = {}
        
        while left < right:
            mid = (left + right) // 2
            memo.clear()
            if canBreakLocks((1 << n) - 1, 0, 1, 0):
                right = mid
            else:
                left = mid + 1
            
        return left