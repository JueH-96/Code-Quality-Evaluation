class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        substring_count = {}
        
        # Count frequency of each k-length substring at valid positions
        for i in range(0, n, k):
            substring = word[i:i+k]
            substring_count[substring] = substring_count.get(substring, 0) + 1
        
        # Find the maximum frequency
        max_frequency = max(substring_count.values())
        
        # Total number of k-length substrings
        total_substrings = n // k
        
        # Minimum operations = total - max_frequency
        return total_substrings - max_frequency