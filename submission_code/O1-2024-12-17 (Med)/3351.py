class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)
        
        max_sum = 0
        # Pick children in sorted order, accounting for the decrement each turn
        for i in range(k):
            # Effective happiness when picked is max(happiness[i] - i, 0)
            curr_happiness = max(happiness[i] - i, 0)
            if curr_happiness == 0:
                # No further picks will contribute if we've reached 0 or below
                break
            max_sum += curr_happiness
        
        return max_sum