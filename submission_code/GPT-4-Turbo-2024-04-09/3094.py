class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        count = Counter(nums)
        operations = 0
        
        # Stores the number of leftover pairs and triplets
        leftover_pairs = 0
        leftover_triplets = 0
        
        for value, freq in count.items():
            pairs, leftover = divmod(freq, 2)
            triplets, leftover = divmod(leftover, 3)
            
            operations += pairs + triplets
            leftover_pairs += pairs
            leftover_triplets += triplets
            
            # Check if the leftover can be combined with previous leftovers
            while leftover_pairs >= 1 and leftover_triplets >= 1:
                # Use one pair and one triplet to form two pairs (4 elements)
                operations += 1
                leftover_pairs -= 1
                leftover_triplets -= 1
        
        # After processing all, check if there are any unpaired leftovers
        if leftover_pairs > 0 or leftover_triplets > 0:
            return -1
        
        return operations