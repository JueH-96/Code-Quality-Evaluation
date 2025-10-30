class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort happiness values in descending order
        happiness.sort(reverse=True)
        
        total_sum = 0
        
        # Select k children
        for i in range(k):
            # The current happiness after i decrements
            current_happiness = happiness[i] - i
            
            # Only add if happiness is positive
            if current_happiness > 0:
                total_sum += current_happiness
            # If current happiness is 0 or negative, all remaining will also be 0 or negative
            # since the array is sorted in descending order
            else:
                break
                
        return total_sum