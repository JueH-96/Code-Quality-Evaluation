class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        from collections import Counter
        MOD = 10**9 + 7
        
        # Count frequency of each number in nums
        freq = Counter(nums)
        
        # We need to find subsequences of length 5 where the middle element is a unique mode
        result = 0
        
        # Iterate through each number to consider it as the middle element of the subsequence
        for middle in freq:
            # The middle element must be the unique mode, so it must appear more than any other element
            # in the subsequence. We need at least 3 occurrences of `middle` to ensure it's the mode.
            if freq[middle] < 3:
                continue
            
            # Choose 3 of `middle` to be in the subsequence
            count_middle = freq[middle]
            choose_3_middle = self.comb(count_middle, 3)
            
            # Now we need to choose 2 more elements from nums excluding these 3 `middle` elements
            remaining_elements = sum(freq.values()) - 3
            
            # Choose any 2 elements from the remaining elements
            if remaining_elements >= 2:
                choose_2_others = self.comb(remaining_elements, 2)
                result += choose_3_middle * choose_2_others
                result %= MOD
        
        return result
    
    def comb(self, n, k):
        # Calculate n choose k
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)  # Take advantage of symmetry
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c