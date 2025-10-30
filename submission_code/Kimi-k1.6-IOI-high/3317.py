class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the frequency of each character
        counts = [0] * 26
        for word in words:
            for c in word:
                counts[ord(c) - ord('a')] += 1
        
        total_pairs = sum(c // 2 for c in counts)
        initial_odd = sum(c % 2 for c in counts)
        
        # Create a list of (required_pairs, is_odd) for each word
        required = []
        for word in words:
            length = len(word)
            req_pairs = length // 2
            is_odd = 1 if length % 2 == 1 else 0
            required.append((req_pairs, is_odd))
        
        # Sort the required list by the number of pairs needed
        required.sort()
        n = len(required)
        
        # Compute prefix sums for pairs and odd counts
        prefix_pairs = [0] * (n + 1)
        prefix_odd = [0] * (n + 1)
        for i in range(n):
            prefix_pairs[i + 1] = prefix_pairs[i] + required[i][0]
            prefix_odd[i + 1] = prefix_odd[i] + required[i][1]
        
        # Check from the largest possible number of palindromes down to 0
        for m in range(n, -1, -1):
            sum_pairs = prefix_pairs[m]
            if sum_pairs > total_pairs:
                continue
            o = prefix_odd[m]
            available_centers = initial_odd + 2 * (total_pairs - sum_pairs)
            if o <= available_centers:
                return m
        
        return 0