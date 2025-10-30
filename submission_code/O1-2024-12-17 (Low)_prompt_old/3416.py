class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert all integers to strings for easy digit comparison
        strs = list(map(str, nums))
        n = len(strs)
        
        # Number of digits in each integer (all have the same length)
        length = len(strs[0])
        
        # We'll accumulate the total sum of digit differences here
        total_differences = 0
        
        # Pre-calculate total pairs once (combination of n items taken 2 at a time)
        total_pairs = n * (n - 1) // 2
        
        # For each digit position, count digit frequency and compute pairs that differ
        for pos in range(length):
            freq = [0]*10
            # Count frequency of digits at this position
            for i in range(n):
                digit = int(strs[i][pos])
                freq[digit] += 1
            
            # Compute the number of pairs with the same digit in this position
            same_pairs = 0
            for d in range(10):
                count_d = freq[d]
                same_pairs += count_d * (count_d - 1) // 2
            
            # The number of pairs that differ at this position
            differences_at_pos = total_pairs - same_pairs
            total_differences += differences_at_pos
        
        return total_differences