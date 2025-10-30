class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort happiness values in descending order
        happiness.sort(reverse=True)
        
        total_happiness = 0
        
        # Select k children
        for i in range(k):
            # Calculate current happiness after i decrements
            current_happiness = max(0, happiness[i] - i)
            total_happiness += current_happiness
            
        return total_happiness