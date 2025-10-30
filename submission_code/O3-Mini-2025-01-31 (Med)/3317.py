from collections import Counter
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Explanation:
        # We are allowed to swap any characters arbitrarily among all words.
        # That means that eventually we can reassign the global multiset of characters
        # (obtained by concatenating all words) arbitrarily into the words (respecting their lengths).
        # We want to maximize the number of words that can be rearranged into a palindrome.
        #
        # A necessary and sufficient condition for a string of length L to be rearranged into a palindrome is:
        #   If L is even: all character counts must be even.
        #   If L is odd: at most one character count can be odd.
        #
        # When we “assign” characters to words to obtain palindromic configurations,
        # each word of odd length will “absorb” one odd character, while an even length word must have 0 odds.
        # Thus if we want to make m words palindromic, the global distribution that goes into those m words 
        # must have exactly "number of odd-length words among those m" letters with odd counts.
        #
        # Given that we can swap arbitrarily, the only thing that might potentially limit the maximum number
        # of palindromes is if the total global pool of characters (the concatenation of all words)
        # has too many letters with odd frequencies.
        # Let X be the number of characters (i.e. letters) whose total count (over all words) is odd.
        #
        # When reassigning the characters among words:
        # • We can “pair up” two odd occurrences (by swapping one from one word to another) so that both
        #   become even – but each such pairing operation can only fix two odd occurrences in the distribution.
        # • However, note that one odd occurrence is allowed overall: For example, if a word is odd-length,
        #   it may have exactly one odd-count letter.
        #
        # Therefore, if we have X odd occurrences in the total multiset,
        # we can “fix” them by pairing them up. Roughly, the maximum number of palindromic words that we can form is:
        #
        #      answer = n - ceil((X - 1) / 2)    if X > 0
        #                 = n                 if X == 0
        #
        # This works because one odd letter can be assigned freely (typically to one odd‐length word).
        # Every pair of extra odd letters (i.e. every two odds beyond that single free one) require that
        # at least one word be “sacrificed” (i.e. not forced into a palindromic arrangement).
        #
        # We compute X by summing, over all letters in the global pool, (frequency % 2).
        # (If a letter appears an even number of times, it’s fine; if odd, we count 1 toward X.)
        #
        # Finally, note that we are free to leave some words non‐palindromic if necessary
        # (and the problem asks for the maximum number of words that can be palindromes after operations).
        
        # Global concatenation of all characters from the words
        total_counter = Counter()
        for word in words:
            total_counter.update(word)
            
        # Count how many letters occur an odd number of times
        odd_count = sum(freq % 2 for freq in total_counter.values())
        
        n = len(words)
        # If odd_count is 0, then we can distribute characters so that every word (even
        # for even-length ones) gets all even counts.
        # Otherwise, one odd can be absorbed without issue, but every pair of extra odd letters
        # effectively forces one word to remain non-palindromic.
        if odd_count <= 1:
            return n
        else:
            # Each pair of excess odd letters (beyond one) costs one word.
            penalty = (odd_count - 1) // 2
            return n - penalty