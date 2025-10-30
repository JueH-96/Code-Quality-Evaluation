class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        
        # Iterate through possible values of 't' (time in seconds).
        # 't' starts from 1 as per the problem's "greater than zero".
        # The loop will find an answer at or before t = ceil(n/k).
        # Since k >= 1, ceil(n/k) is at most n. So, iterating up to n is sufficient.
        for t in range(1, n + 1): 
            # Calculate the total number of characters that would have been removed
            # from the prefix of the original word after 't' seconds.
            current_removed_chars_count = t * k
            
            # Condition Check:
            # We want to find the minimum 't' where the word can revert to its initial state.
            # This happens if the portion of the original word that remains after 't*k' removals
            # matches the corresponding prefix of the original word.
            
            # Case 1: All original characters (or more) have been removed.
            # If current_removed_chars_count >= n, it means we've effectively cleared the word
            # of its original content. In this scenario, the remaining part of the original string
            # (word[current_removed_chars_count:]) is an empty string, and the target prefix
            # (word[:n - current_removed_chars_count]) is also an empty string.
            # Since we can add *any* characters, we can simply add the entire original 'word'
            # to revert to its initial state. This is a valid time 't'.
            if current_removed_chars_count >= n:
                return t
            
            # Case 2: Some original characters still remain.
            # We check if the suffix of the original word starting at current_removed_chars_count
            # (i.e., word[current_removed_chars_count:]) is identical to the prefix of the original word
            # of the same length (i.e., word[:n - current_removed_chars_count]).
            # If they match, it means the structure allows for the word to revert.
            # The remaining part of the original word forms the required prefix.
            # We can then choose the characters to add to the end to form the required suffix
            # of the initial word (specifically, word[n - current_removed_chars_count:]).
            if word[current_removed_chars_count:] == word[:n - current_removed_chars_count]:
                return t
        
        # This line should theoretically never be reached because the loop covers all
        # possibilities up to t=n, and at t=ceil(n/k) (which is <= n), a solution is guaranteed.
        # It's included for completeness, but indicates a logical flaw if it is ever hit.
        return n