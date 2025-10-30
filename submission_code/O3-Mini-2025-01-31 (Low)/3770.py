class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1
        
        # word: list of characters, initially None
        word = [None] * L
        # fixed[i] will be True if that letter is forced by a "T" constraint.
        fixed = [False] * L
        
        # Step 1: Apply "T" conditions
        for i in range(n):
            if str1[i] == 'T':
                # This forces that for j in [0, m-1], word[i+j] == str2[j]
                if i + m > L:
                    return ""
                for j in range(m):
                    pos = i + j
                    letter = str2[j]
                    if word[pos] is None:
                        word[pos] = letter
                        fixed[pos] = True
                    else:
                        # already assigned: if conflict, then impossible.
                        if word[pos] != letter:
                            return ""
                        # else: already equals letter, so fine.
        
        # For every free position, assign the default letter 'a'
        for i in range(L):
            if word[i] is None:
                word[i] = 'a'
        
        # Helper: Check if all 'F' conditions are satisfied.
        # For every i with str1[i]=='F', the substring word[i:i+m] must NOT equal str2.
        def violates_F_conditions():
            for i in range(n):
                if str1[i] == 'F':
                    # Check substring word[i:i+m] equality with str2.
                    # (We already have word fully assigned.)
                    # If the substring equals str2 exactly, then violation.
                    # Use simple iteration because m<=500.
                    good = False
                    for j in range(m):
                        if word[i + j] != str2[j]:
                            good = True
                            break
                    if not good:
                        return i  # Return the starting index of the violating window.
            return -1
        
        # Function to fix a violation for window starting at index win_start.
        # We try to bump one free position (not fixed) in the window [win_start, win_start+m-1].
        # We choose the rightmost free position available (to keep prefix small).
        def fix_violation(win_start: int) -> bool:
            # We go in reverse order so that we change as far right as possible.
            for j in range(m-1, -1, -1):
                pos = win_start + j
                # We can only change if the position is not fixed by T.
                if not fixed[pos]:
                    current = word[pos]
                    # Now, we must choose a letter greater than 'current' up to 'z'
                    # but also we must avoid making it exactly equal to str2[j]
                    for asc in range(ord(current) + 1, ord('z') + 1):
                        candidate = chr(asc)
                        # If candidate letter equals the corresponding letter of str2, skip; 
                        # because if we set it equal then the whole window might still match.
                        if candidate == str2[j]:
                            continue
                        # We found a candidate for pos.
                        word[pos] = candidate
                        # For every free position > pos, we reset them to 'a' (lowest possible)
                        for k in range(pos + 1, L):
                            if not fixed[k]:
                                word[k] = 'a'
                        return True
            return False  # Could not fix violation.
        
        # Main loop: try to fix any F violation step by step.
        while True:
            viol_index = violates_F_conditions()
            if viol_index == -1:
                break  # All F conditions satisfied.
            # Try to fix the violation window starting at viol_index.
            if not fix_violation(viol_index):
                return ""
            # Continue; the fix may have fixed many windows, but we recheck.
        
        return "".join(word)


# Optional test cases. You can run the following to test the solution.
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.generateString("TFTF", "ab"))   # Expected output: "ababa"
    # Example 2:
    print(sol.generateString("TFTF", "abc"))  # Expected output: ""
    # Example 3:
    print(sol.generateString("F", "d"))       # Expected output: "a"