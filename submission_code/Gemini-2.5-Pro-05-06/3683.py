class Solution:
  def answerString(self, word: str, numFriends: int) -> str:
    N = len(word)
    
    # Calculate the maximum allowed length for any single part of the split.
    # If a part has length L_s, the other (numFriends - 1) parts must sum to N - L_s.
    # Each of these (numFriends - 1) parts must be non-empty (length >= 1).
    # So, N - L_s >= numFriends - 1.
    # This implies L_s <= N - (numFriends - 1) = N - numFriends + 1.
    max_len_allowed = N - numFriends + 1
    
    # Initialize max_s (the lexicographically largest string found so far).
    # An empty string is lexicographically smaller than any non-empty string.
    # All valid parts are non-empty.
    max_s = "" 

    # Iterate over all possible start indices `i` for a substring.
    for i in range(N):
        # Optimization:
        # If max_s has been found and its first character is greater than word[i],
        # then any substring starting with word[i] will be lexicographically smaller
        # than max_s. So, we can skip all substrings starting at index `i`.
        if max_s != "" and word[i] < max_s[0]:
            continue
        
        # If word[i] == max_s[0], we might still find a better string if the
        # current character matches max_s's first character but subsequent characters are larger,
        # or if it's a longer string that starts with the same prefix.
        # e.g., max_s = "ax", current candidate word[i:...] could be "ay..." or "axb..."
        # If word[i] > max_s[0] (and max_s is not empty), or if max_s is empty,
        # any substring starting at word[i] has the potential to become the new max_s.

        # Determine the upper bound for the length of substrings starting at i.
        # Max possible length for a substring starting at i is N - i.
        # It must also be at most max_len_allowed.
        upper_bound_for_length_at_i = min(max_len_allowed, N - i)

        for length in range(1, upper_bound_for_length_at_i + 1):
            # The current substring being considered is word[i : i + length]
            
            # Another small optimization check before slicing:
            # If word[i] == max_s[0] (we already know word[i] >= max_s[0] from above check, if max_s non-empty)
            # AND current length is less than len(max_s)
            # AND the first char of current_substring (word[i]) is same as max_s[0]
            # AND current_substring (word[i:i+length]) is a prefix of max_s
            # Then current_substring is smaller than max_s. e.g. max_s="apple", current="app"
            # This check is implicitly handled by `current_substring > max_s` and might
            # not save much if slicing is the main bottleneck. But let's be explicit if it helps.
            # Consider: max_s = "banana", current_start_char = 'b'.
            # length=1: "b".  If current_substring > max_s: update. ("b" vs "banana" -> no)
            # length=2: "ba". If current_substring > max_s: update. ("ba" vs "banana" -> no)
            # ...
            # length=6: "banana". If current_substring > max_s: update. ("banana" vs "banana" -> no)
            # This check is only useful if we can avoid the slice for comparison, which is hard.
            # The direct comparison `current_substring > max_s` is generally fine.

            current_substring = word[i : i + length]
            
            # If current_substring is lexicographically greater than max_s, update max_s.
            if current_substring > max_s:
                max_s = current_substring
                
    return max_s