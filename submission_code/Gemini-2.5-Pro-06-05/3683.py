class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        
        # A substring `s` can be part of a split into `numFriends` parts if the
        # rest of the word, `n - len(s)` characters, can form `numFriends - 1`
        # non-empty parts. This requires `n - len(s) >= numFriends - 1`.
        # Thus, the maximum length of a valid substring is `n - numFriends + 1`.
        max_len = n - numFriends + 1
        
        # The problem is now to find the lexicographically largest substring of `word`
        # with a length no more than `max_len`.
        
        max_substring = ""
        
        # Iterate through all possible starting positions `i` for a substring.
        for i in range(n):
            # For a fixed starting position `i`, the longest valid substring is also
            # the lexicographically largest among all valid substrings starting at `i`.
            # This is because in lexicographical order, "a" < "ab", "abc" < "abcd", etc.
            end = min(i + max_len, n)
            candidate = word[i:end]
            
            # Compare the current candidate with the largest one found so far.
            if candidate > max_substring:
                max_substring = candidate
                
        return max_substring