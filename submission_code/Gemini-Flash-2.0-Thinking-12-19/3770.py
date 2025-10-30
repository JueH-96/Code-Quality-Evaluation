class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        word_len = n + m - 1
        word = [''] * word_len # Using list of characters for mutability

        for j in range(word_len):
            # 1. Determine required character for word[j] from 'T' conditions
            required_char_t = None
            t_requirements = set()
            
            # Iterate through all indices i in str1 that affect word[j]
            # word[j] is in word[i:i+m] if i <= j < i + m, which means j - m < i <= j
            # Combined with 0 <= i < n, the range for i is max(0, j - m + 1) <= i <= min(n - 1, j)
            # Python range is exclusive upper bound, so goes up to min(n - 1, j) + 1
            # min(n - 1, j) + 1 == min(n, j + 1)
            for i in range(max(0, j - m + 1), min(n, j + 1)):
                 if str1[i] == 'T':
                    # If str1[i] is 'T', the substring word[i:i+m] must be str2.
                    # word[j] is the (j-i)-th character within this substring.
                    # So, word[j] must be str2[j - i].
                    t_requirements.add(str2[j - i])

            if len(t_requirements) > 1:
                # Conflicting 'T' requirements for word[j] from different i indices
                return "" # No string satisfying the conditions can be generated.

            if len(t_requirements) == 1:
                # 'T' condition(s) uniquely determine the character for word[j]
                char_to_set = list(t_requirements)[0]
            else:
                # No 'T' condition dictates word[j]. We can choose greedily.
                # The smallest possible character is 'a'.
                # However, we must check if choosing 'a' would violate an 'F' condition
                # that is finalized at this step j.
                
                # An 'F' condition str1[i] == 'F' is finalized at index j if j is the last character
                # of the substring word[i:i+m], i.e., j = i + m - 1, which means i = j - m + 1.
                
                char_to_avoid = None
                i_finalizing_f = j - m + 1 # Potential index i for a finalizing 'F' condition
                
                # Check if there is a relevant 'F' condition finalizing at j
                if 0 <= i_finalizing_f < n and str1[i_finalizing_f] == 'F':
                    # The condition is word[i_finalizing_f : i_finalizing_f + m] != str2.
                    # This substring is word[i_finalizing_f : j + 1].
                    # At step j, we are setting word[j], the last character of this substring.
                    # If the prefix word[i_finalizing_f : j] is already equal to str2[0 : m-1],
                    # then setting word[j] to str2[m-1] would make the whole substring equal to str2,
                    # violating the 'F' condition.
                    
                    prefix_match = True
                    # Compare word[i_finalizing_f + k] with str2[k] for k in [0, m-2]
                    # The indices i_finalizing_f + k for k < m-1 are always less than j,
                    # so their values in `word` have already been set in previous iterations.
                    for k in range(m - 1):
                        if word[i_finalizing_f + k] != str2[k]:
                            prefix_match = False
                            break
                            
                    if prefix_match:
                        # If the prefix matches str2[0 : m-1], we must avoid setting word[j] to str2[m-1]
                        char_to_avoid = str2[m - 1]

                # Choose the smallest possible character for word[j].
                # Start with 'a'.
                char_to_set = 'a'
                # If 'a' is the character we must avoid because it would violate
                # the finalizing 'F' condition, choose the next smallest character 'b'.
                if char_to_avoid is not None and char_to_set == char_to_avoid:
                    char_to_set = 'b'
                    # Note: We only need to check 'a'. If 'a' is not forbidden, we choose it
                    # as it's the lexicographically smallest. If 'a' is forbidden, the next
                    # smallest is 'b'. 'b' cannot also be forbidden by this *same* 'F' condition
                    # because char_to_avoid is a single character str2[m-1].

            # Set the determined character at the current position j
            word[j] = char_to_set

            # 2. After setting word[j], check if any *finalized* 'F' condition is violated
            # The only 'F' condition that is finalized at index j is at i = j - m + 1.
            # We need to re-check this condition now that word[j] is set,
            # regardless of whether word[j] was forced by a 'T' or chosen greedily.
            i_finalizing_f = j - m + 1
            
            # Check if there is a relevant 'F' condition finalizing at j
            if 0 <= i_finalizing_f < n and str1[i_finalizing_f] == 'F':
                 # The condition is word[i_finalizing_f : i_finalizing_f + m] != str2.
                 # This substring is word[i_finalizing_f : j + 1].
                 # Check if the completed substring word[i_finalizing_f : j + 1] is equal to str2.
                 is_str2_match = True
                 # Compare word[i_finalizing_f + k] with str2[k] for k in [0, m-1].
                 # These indices i_finalizing_f + k go up to (j - m + 1) + m - 1 = j.
                 # Accessing word[j] is valid as it was just set.
                 for k in range(m):
                     if word[i_finalizing_f + k] != str2[k]:
                         is_str2_match = False
                         break
                         
                 if is_str2_match:
                      # F condition str1[i_finalizing_f] == 'F' is violated because word[i_finalizing_f : i_finalizing_f + m] == str2
                      return "" # No string satisfying the conditions can be generated.

        # If the loop completes without returning "", a valid string was constructed. Return it.
        return "".join(word)