class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        
        # We need to find the minimum time t > 0. We check t = 1, 2, 3, ...
        # At time t, the prefix of the current word is what remains of the original,
        # which is `word[t*k:]`. We can revert to the initial state if this remaining
        # part is a prefix of the original `word`.
        
        t = 1
        while t * k < n:
            # Check if the suffix of `word` starting at index `t*k` is a prefix of `word`.
            # The `startswith()` method is perfect for this check.
            if word.startswith(word[t * k:]):
                return t
            
            # If no match, advance to the next second.
            t += 1
            
        # If the loop finishes, it means for all t where t*k < n, no match was found.
        # The loop terminates when t is the smallest integer such that t*k >= n.
        # At this time, `word[t*k:]` is empty, which is a prefix of any string,
        # guaranteeing a solution. This value of t is the answer.
        return t