class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1  # length of the generated word
        
        # Initialize candidate word as a list of characters and fixed array.
        candidate = [None] * L
        fixed = [False] * L
        
        # First, propagate T-constraints.
        for i in range(n):
            if str1[i] == 'T':
                # The substring from index i (of length m) must equal str2.
                if i + m > L:  # not enough space, though by problem spec it should.
                    return ""
                for j in range(m):
                    pos = i + j
                    c = str2[j]
                    if candidate[pos] is not None:
                        # Already assigned; it must match.
                        if candidate[pos] != c:
                            return ""
                    candidate[pos] = c
                    fixed[pos] = True
        # For positions not fixed, assign "a" (the lexicographically smallest letter).
        for i in range(L):
            if candidate[i] is None:
                candidate[i] = 'a'
        
        # Now, check every F-condition.
        # For an F-condition at index i, the window candidate[i:i+m] must NOT equal str2.
        # But note: forced positions (from T) always equal str2.
        # The only potential freedom is in free positions.
        # With our default assignment (free positions set to "a"), the substring equals str2
        # if for every j from 0 to m-1:
        #   - if the position is free then since candidate is "a" it must be that str2[j]=="a"
        #   - if the position is forced, it is equal to str2[j] (by construction).
        # So the F-condition is violated exactly when for all j in 0...m-1 where (i+j) is free, we have str2[j]=="a".
        # For each such violation, we need to choose a free position in the window and change it (upgrade "a"->"b").
        violation_records = []  # list of tuples: (i, allowed_set, R)
        for i in range(n):
            if str1[i] == 'F':
                # Check window candidate[i:i+m]
                window_violation = True
                free_found = False  # has at least one free position?
                # allowed set: positions which are free (not fixed) and (by default "a") that correspond to str2 letter "a"
                allowed = []
                for j in range(m):
                    pos = i + j
                    if not fixed[pos]:
                        free_found = True
                        if str2[j] != 'a':
                            # Even though pos is free, its default letter "a" differs from str2[j].
                            window_violation = False
                            break
                        else:
                            allowed.append(pos)
                    # If fixed, then candidate[pos]==str2[j] by construction.
                # If no free positions exist then the window is completely forced.
                # In that case, candidate[i:i+m] == str2 (because forced positions equal str2) 
                # and the F condition is impossible to satisfy.
                if not free_found:
                    # The window is fully determined by a T constraint and equals str2.
                    return ""
                if window_violation:
                    # record the allowed free positions (as a set for fast membership checks)
                    # and also record the rightmost free position among them (we want to fix as late as possible).
                    r = max(allowed)
                    violation_records.append( (i, set(allowed), r) )
        
        # Now, we need to choose a set S (subset of free positions) so that for every violation record (i, X_i, R_i),
        # S intersects X_i. To keep the overall candidate as lexicographically small as possible,
        # we want the first (smallest index) change to be as far to the right as possible.
        # A greedy strategy works: sort the violation records by R_i in increasing order,
        # and if the current violation is not “covered” (i.e. none of the chosen positions is in the allowed set),
        # then choose the free position equal to R_i (the rightmost possible free position in that window).
        violation_records.sort(key=lambda x: x[2])  # sort by R (the rightmost free pos in that window)
        chosen = set()
        for rec in violation_records:
            (_, X, R) = rec
            # Check if current violation is already fixed by a chosen free position.
            # i.e. if chosen ∩ X is nonempty.
            covered = False
            for p in chosen:
                if p in X:
                    covered = True
                    break
            if not covered:
                # choose R to cover this violation.
                chosen.add(R)
        
        # For each free position in chosen, update candidate by changing the letter "a" to "b"
        # (which is the smallest letter not equal to "a")
        for p in chosen:
            candidate[p] = 'b'
        
        # (No further adjustments are necessary because after these changes every F-condition is now satisfied.)
        # Finally, double-check the F–conditions if you wish (optional) – but here we assume correctness.
        # Return the generated word.
        return "".join(candidate)