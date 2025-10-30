class Solution:
  def generateString(self, str1: str, str2: str) -> str:
    """
    Generates the lexicographically smallest string based on the given constraints.

    The method constructs the target string `word` of length `n + m - 1` by first satisfying
    the most restrictive constraints ('T' conditions), then greedily filling the rest to
    form a base candidate, and finally adjusting this candidate to satisfy the 'F'
    conditions while maintaining the lexicographically smallest property.
    """
    n = len(str1)
    m = len(str2)
    L = n + m - 1
    
    word = [''] * L
    is_fixed = [False] * L
    
    # Step 1: Populate based on 'T' constraints
    for i in range(n):
        if str1[i] == 'T':
            for j in range(m):
                pos = i + j
                char_to_set = str2[j]
                if word[pos] != '' and word[pos] != char_to_set:
                    return ""  # Conflict
                word[pos] = char_to_set
                is_fixed[pos] = True

    # Step 2: Check for impossible 'F' constraints violated by 'T's alone
    for i in range(n):
        if str1[i] == 'F':
            is_violated_by_Ts = True
            for j in range(m):
                pos = i + j
                # If a position is not fixed by a 'T', or if it is fixed but to a different character,
                # then this 'F' constraint is not (yet) violated by 'T' constraints alone.
                if not is_fixed[pos] or word[pos] != str2[j]:
                    is_violated_by_Ts = False
                    break
            if is_violated_by_Ts:
                return ""

    # Step 3: Greedy construction - start with the lexicographically smallest candidate
    final_word = list(word)
    for k in range(L):
        if final_word[k] == '':
            final_word[k] = 'a'

    # Step 4: Fix 'F' violations by making minimal lexicographical changes
    for i in range(n):
        if str1[i] == 'F':
            is_match = True
            for j in range(m):
                if final_word[i + j] != str2[j]:
                    is_match = False
                    break
            
            if is_match:
                # Violation found. Fix it by changing the rightmost possible character.
                k_to_change = -1
                for j in range(m - 1, -1, -1):
                    k = i + j
                    if not is_fixed[k] and final_word[k] != 'z':
                        k_to_change = k
                        break
                
                if k_to_change == -1:
                    # All modifiable characters in the slice are 'z'.
                    # Cannot break the match by incrementing.
                    return ""

                # Increment the character to break the match.
                final_word[k_to_change] = chr(ord(final_word[k_to_change]) + 1)
    
    return "".join(final_word)