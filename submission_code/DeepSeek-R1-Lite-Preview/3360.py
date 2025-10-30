class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Step 1: Count frequency of each character
        freq = {}
        for char in word:
            if char in freq:
                freq[char] += 1
            else:
                freq[char] = 1
        
        # Step 2: Sort the frequencies
        sorted_freq = sorted(freq.values())
        
        # Step 3: Use sliding window to find maximum sum of frequencies where max - min <= k
        n = len(sorted_freq)
        max_sum = 0
        current_sum = 0
        left = 0
        
        for right in range(n):
            current_sum += sorted_freq[right]
            # While the difference exceeds k, move the left pointer
            while sorted_freq[right] - sorted_freq[left] > k:
                current_sum -= sorted_freq[left]
                left += 1
            # Update the maximum sum
            if current_sum > max_sum:
                max_sum = current_sum
        
        # Step 4: Calculate minimal deletions
        total_freq = sum(sorted_freq)
        min_deletions = total_freq - max_sum
        return min_deletions