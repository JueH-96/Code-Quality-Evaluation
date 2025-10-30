from collections import Counter
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Explanation:
        # We are allowed to swap any characters between any words.
        # However, we cannot change the word lengths.
        # Thus, our “resources” are the overall counts of letters, which are fixed.
        #
        # To arrange a word into a palindrome:
        # • For an even-length word, every letter used must appear an even number of times.
        # • For an odd-length word, exactly one letter may appear an odd number of times and 
        #   the rest must appear in even numbers.
        #
        # When we have complete freedom of reassignment (swaps), we want to “build” palindromic
        # versions of as many words as possible, by partitioning the overall collection of letters
        # among words in a way that fits the requirement.
        #
        # Think of it this way:
        #   For a word of length L, a palindrome must allocate:
        #     pairs_required = L // 2   (each pair contributes 2 letters)
        #     single_needed = (L % 2)    (center letter for odd-length words)
        #
        # However, the available letters come in two “forms,” effectively:
        #   - Pairs available from each letter: for a letter with count c, it has c//2 pairs.
        #   - Singletons: the leftover count (c % 2).
        #
        # In total, the number of pairs available is fixed as:
        #     total_pairs = sum( c // 2 for each letter )
        # and the number of singletons available is:
        #     odd_total = sum( c % 2 for each letter )
        #
        # But even if there is a deficit of singletons to use as "center letters" for odd-length words,
        # a pair can be “broken” into two singles. Breaking a pair uses up one pair and yields 2 singles.
        #
        # Let:
        #    #words = n
        #    odd_words = count of words with odd length (each requires 1 single)
        #
        # To form palindromes in all words, we would need at least odd_words center letters.
        # We already have odd_total singles available. If odd_total is less than odd_words,
        # we can break some pairs into singles. But if we have more singles than odd_words,
        # then the “excess” singles cannot help directly because extra singles in an even-length word
        # would violate the palindrome structure.
        #
        # In fact, one cannot “create” palindromic structure if the number of available singles (possibly
        # after pair-breaking) is more than the number of odd length words. From an opposite point of view,
        # if there is an "excess" of singles beyond the odd_words, then some words will be forced to use
        # a pair (by pairing two singles together) instead of having a nice center.
        #
        # It turns out – and this is the key observation – that the maximum number of words we can convert
        # into palindromes is given by:
        #
        #     answer = n - max(0, (odd_total - odd_words) // 2)
        #
        # Why? Each 2 "excess" singles above those needed for odd-length words forces us to “lose” a potential
        # palindrome because those extra singles require pairing up (thus “fixing” an odd word into an even format,
        # which is impossible if the word originally had an odd length requirement).
        #
        # Let’s check with one of the examples:
        #   Example 3: words = ["cd","ef","a"]
        #     Global letters counts: {'c':1, 'd':1, 'e':1, 'f':1, 'a':1}
        #     odd_total = 5  (all letters appear once)
        #     odd_words = 1  (only "a" has odd length)
        #
        #     extra = odd_total - odd_words = 4, so extra//2 = 2.
        #     answer = n - 2 = 3 - 2 = 1, which matches the example.
        #
        # Similarly, for Example 1:
        #   words = ["abbb","ba","aa"]
        #     Global counts: {'a':4, 'b':4} so odd_total = 0;
        #     odd_words = 0 (all words have even length)
        #     answer = n - max(0, (0 - 0)//2) = 3.
        #
        # And for Example 2:
        #   words = ["abc","ab"]
        #     Global counts: {'a':2, 'b':2, 'c':1} so odd_total = 1;
        #     odd_words = 1 (only "abc" is odd-length)
        #     answer = 2 - max(0, (1 - 1)//2) = 2.
        #
        # This reasoning leads to the simple formula:
        #
        #   max_palindromes = n - max(0, (odd_total - odd_words) // 2)
        #
        # Note: In all cases, the answer cannot exceed n.
        
        n = len(words)
        # Count of words that require a center letter (odd-length words)
        odd_words = sum(1 for word in words if len(word) % 2 == 1)
        
        # Build a global counter for all letters across all words.
        counter = Counter()
        for word in words:
            counter.update(word)
        
        # Count total singles (i.e. letters that contribute an odd count)
        odd_total = sum(count % 2 for count in counter.values())
        
        # Every two extra singles (above what odd_words demand) will force us to ruin one palindrome.
        penalty = 0
        if odd_total > odd_words:
            penalty = (odd_total - odd_words) // 2
        return n - penalty