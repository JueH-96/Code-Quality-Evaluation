class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)

        # Handle edge case k=0, always true.
        if k == 0:
            return True
        
        # If string length is 1 and k > 0, cannot form any special substring (condition 2: not entire string)
        # s.length >= 2, so n >= 2 is guaranteed.
        # However, if k > 0 and no special substrings can be found, it's False.

        # 1. Precompute prefix and suffix character sets
        # prefix_chars[p] = set of unique characters in s[0:p]
        prefix_chars = [set()] * (n + 1)
        current_set = set()
        for i in range(n):
            current_set.add(s[i])
            prefix_chars[i+1] = current_set.copy() # Store a copy

        # suffix_chars[p] = set of unique characters in s[p:n]
        suffix_chars = [set()] * (n + 1)
        current_set = set()
        for i in range(n - 1, -1, -1):
            current_set.add(s[i])
            suffix_chars[i] = current_set.copy() # Store a copy
        
        # If a character does not appear after index n-1, suffix_chars[n] will be empty.
        # This is handled correctly by the loop. suffix_chars[n] should indeed be empty.
        suffix_chars[n] = set()

        # 2. Identify all possible single-character special substrings
        # A substring s[p:p+1] is special if:
        # 1. {s[p]} is disjoint from (set(s[0:p]) union set(s[p+1:n]))
        # 2. s[p:p+1] is not the entire string s (length 1 < n if n >= 2)
        
        candidate_intervals = []
        for p in range(n):
            # Condition 2: Check if s[p:p+1] is the entire string
            # This is only possible if n=1. But problem states n >= 2. So length 1 substring is never the entire string.
            
            char_in_sub = {s[p]}
            chars_outside_sub = prefix_chars[p].union(suffix_chars[p+1])

            if char_in_sub.isdisjoint(chars_outside_sub):
                candidate_intervals.append((p, p + 1)) # (start_index, end_index_exclusive)

        # 3. Maximize disjoint intervals
        # Sort candidate_intervals by their end points
        candidate_intervals.sort(key=lambda x: x[1])

        max_disjoint_count = 0
        last_end_picked = -1 # Represents the exclusive end index of the last chosen interval

        for start, end in candidate_intervals:
            if start >= last_end_picked:
                max_disjoint_count += 1
                last_end_picked = end
        
        return max_disjoint_count >= k