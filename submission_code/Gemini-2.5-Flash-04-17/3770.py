class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1

        # word_chars stores the characters of the generated string.
        # Initially, all positions are None, meaning they are not fixed by 'T' constraints yet.
        word_chars = [None] * L

        # Step 2: Process 'T' constraints and check conflicts
        # If str1[i] is 'T', the substring word[i:i+m] must be equal to str2.
        # This fixes the characters in word_chars[i] through word_chars[i+m-1].
        # Check for conflicts if a position is required to be different characters by different 'T' constraints.
        for i in range(n):
            if str1[i] == 'T':
                for j in range(m):
                    p = i + j
                    # The index p = i + j will always be within the bounds [0, L-1]
                    # because 0 <= i <= n-1 and 0 <= j <= m-1, so 0 <= i+j <= n-1+m-1 = L-1.
                    
                    if word_chars[p] is None:
                        word_chars[p] = str2[j]
                    elif word_chars[p] != str2[j]:
                        # Conflict found: position p is required to be word_chars[p]
                        # by a previous 'T' constraint, but now required to be str2[j].
                        return "" # No such string can be generated

        # Step 3: Determine character requirement ('b' if 'a' is forbidden by 'F')
        # An 'F' constraint str1[i] == 'F' requires word[i:i+m] != str2.
        # If an 'F' constraint is not already satisfied by characters fixed by 'T' constraints,
        # we must make a change at some position within the window [i, i+m-1].
        # To find the lexicographically smallest string, this change should occur at the
        # rightmost possible index within the window that is not fixed by 'T'.
        # If setting this rightmost changeable position to 'a' would result in the
        # entire window matching str2, we must set it to 'b' instead.
        
        # required_char[k] will be 'b' if the character at index k *must* be 'b' (or higher)
        # to satisfy some 'F' constraint, based on the rightmost changeable position logic.
        # Initially all requirements are None, meaning the default smallest character 'a' is preferred.
        required_char = [None] * L 

        for i in range(n):
            if str1[i] == 'F':
                # Check if constraint `i` is already satisfied by fixed characters.
                # An 'F' constraint is satisfied if there exists at least one position p
                # in the window [i, i+m-1] where word_chars[p] is fixed by 'T' AND word_chars[p] != str2[p-i].
                already_satisfied = False
                for j in range(m):
                    p = i + j
                    # p is always in [0, L-1]
                    
                    if word_chars[p] is not None: # This position is fixed by 'T'
                        if word_chars[p] != str2[j]:
                            already_satisfied = True
                            break # This F constraint is satisfied by a fixed non-match

                if already_satisfied:
                    continue # This F constraint is satisfied, no need to force changes for it

                # Constraint `i` is not satisfied by fixed characters.
                # This means for all j in [0, m-1], word_chars[i+j] is either None OR word_chars[i+j] == str2[j].
                
                # Find the rightmost index `k` in `[i, i+m-1]` where `word_chars[k]` is `None`.
                # This is the rightmost position we are free to change to satisfy this constraint.
                rightmost_changeable_k = -1
                # Iterate from right to left within the window [i, i+m-1]
                # The range [i, i+m-1] is fully contained within [0, L-1] because 0 <= i < n and 1 <= m.
                for k_check in range(i + m - 1, i - 1, -1): 
                    if word_chars[k_check] is None: # This position is not fixed by 'T', so we can potentially change it.
                        rightmost_changeable_k = k_check
                        break

                # If rightmost_changeable_k == -1, it means all positions in [i, i+m-1] were fixed by 'T'.
                # Since `already_satisfied` was False, this implies word_chars[i+j] == str2[j] for all j.
                # This forces the substring word[i:i+m] to be equal to str2, which conflicts with str1[i] == 'F'.
                # Therefore, no such string can be generated.
                if rightmost_changeable_k == -1:
                    return ""

                # We found the rightmost changeable index `k = rightmost_changeable_k` for constraint `i`.
                # To satisfy this constraint, word[k] must be different from str2[k-i].
                # If str2[k-i] is 'a', then setting word[k] to 'a' would make word[k] match str2[k-i].
                # If setting word[k] to 'a' would cause the *entire* window word[i:i+m] to match str2
                # (given that other changeable positions will default to 'a' and fixed positions match),
                # then we *must* set word[k] to 'b' to break the match.
                # This condition holds true precisely when str2[k-i] is 'a', because k is the rightmost
                # changeable position, meaning all positions p > k in the window are fixed and match str2,
                # and all positions p < k in the window are either fixed and match str2, or are changeable
                # and will be set to 'a' which we assume matches str2[p-i] for this problematic case.
                # Thus, if str2[k-i] is 'a', we *must* make word[k] at least 'b' to satisfy constraint `i`.
                # Mark this requirement at index k. If multiple constraints require 'b' at k, it's consistent.
                if str2[rightmost_changeable_k - i] == 'a':
                    required_char[rightmost_changeable_k] = 'b'

        # Step 4: Construct the result string by filling in None's
        # Iterate through all positions k from 0 to L-1.
        # If word_chars[k] is still None, it means it was not fixed by a 'T' constraint.
        # We fill it with the smallest possible character, which is 'a', unless a required_char
        # was marked as 'b' during the 'F' constraint processing.
        for k in range(L):
            if word_chars[k] is None: # This position was not fixed by 'T'
                if required_char[k] == 'b':
                    word_chars[k] = 'b' # Forced to be 'b' by an 'F' constraint
                else:
                    word_chars[k] = 'a' # Default smallest character 'a'

        # Step 5: Convert the list of characters to a string and return.
        return "".join(word_chars)