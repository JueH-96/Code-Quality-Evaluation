class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort in descending order so we always pick the largest remaining happiness first
        happiness.sort(reverse=True)
        
        total = 0
        # When picking the i-th child (0-indexed), its value has been decremented i times
        for i in range(k):
            contrib = happiness[i] - i
            if contrib <= 0:
                # Once contributions hit zero or below, no further picks add anything
                break
            total += contrib
        
        return total