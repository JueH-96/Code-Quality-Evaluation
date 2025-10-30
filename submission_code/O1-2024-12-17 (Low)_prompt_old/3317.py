class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        from collections import Counter
        
        # Gather the total frequency of all characters across all words
        freq = Counter()
        for w in words:
            freq.update(w)
        
        # Total number of pairs we can form
        total_pairs = sum(v // 2 for v in freq.values())
        # Total number of single leftover chars (potential centers for odd-length palindromes)
        total_singles = sum(v % 2 for v in freq.values())
        
        # Collect lengths of all words
        lengths = [len(w) for w in words]
        # Sort lengths by the number of needed pairs ascending,
        # and if there's a tie in pairs needed, place even-length first
        lengths.sort(key=lambda x: (x // 2, x % 2))
        
        count_palindromes = 0
        
        for length in lengths:
            needed_pairs = length // 2
            needed_singles = length % 2  # 1 if odd, 0 if even
            # Check feasibility
            if total_pairs >= needed_pairs and total_singles >= needed_singles:
                # We can form this palindrome
                total_pairs -= needed_pairs
                total_singles -= needed_singles
                count_palindromes += 1
            # Otherwise, we skip it and try the next (which might need same or more pairs)
        
        return count_palindromes