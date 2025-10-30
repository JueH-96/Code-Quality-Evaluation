class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        """
        We want the smallest positive T so that after T "remove k + append k" operations
        (with the appended k letters chosen arbitrarily each time), the string returns
        to its original value.

        Key idea:
          • After each step we cut off the first k characters and then append k new ones.
          • The only part we cannot freely alter is the 'middle' n-k characters that get
            carried over from the previous step (positions [k..n-1] of the old word
            become positions [0..n-k-1] of the new word). 
          • Appended characters (the last k of the new word) can be chosen arbitrarily to
            match whatever we need.
          • Hence, the only real "constraint" is how characters propagate from one step
            to the next through the first n-k positions.

        A convenient way to check feasibility for a given T:
          • We require that the final word (after T steps) match the original word.
          • For each final position j (0 <= j < n), we ask: does the character in position j
            at step T come from some old position of the string (that we couldn't freely set),
            or does it come from the appended region (which we can set to anything)?
          • At a given step i, positions [0..n-k-1] of that step's word are "inherited"
            from positions [k..n-1] of step (i-1)'s word, while positions [n-k..n-1]
            are freely chosen (appended).
          • We can trace each final position j backward through the steps, following
            the "inherited" slice as long as j < n-k and we haven't run out of steps.
            Each time we go back a step i → i-1, the inherited position is j+k in the old word.
            If at some step this index j+k >= n-k (meaning it falls in the appended region),
            we stop tracing further because that character could have been chosen arbitrarily.
          • If we do manage to trace all the way back i steps (i.e. from step T down to step 0)
            without entering the appended region, that means the final character at position j
            was forced to match the original word's character at some specific position.
            This gives a constraint word[original_position] == word[j].
          • If any of these forced constraints conflict, that T is not possible.
            Otherwise it is possible (since the appended parts can be anything we need).
        
        We then try T = 1..n (in the worst case T = n will remove the entire string after
        enough steps, allowing us to rebuild it freely) and return the first feasible T.
        """

        n = len(word)
        # Try each T from 1 to n inclusive.
        for T in range(1, n+1):
            # We will collect constraints of the form:
            #   "word[pos_in_original] must equal word[pos_in_original_again]"
            # in a dictionary constraints: pos -> required_letter
            # If there's a conflict, we know T is not possible.
            forced = {}
            conflict = False

            for j in range(n):
                # We'll trace back from (T, j) while j < n-k.
                # i goes from T down to 0 (the original string is "step 0"),
                # pos starts at j and moves by +k each step back.
                i = T
                pos = j
                # Trace back until we either run out of steps (i=0) or
                # we find pos >= n-k (meaning that character was appended freely).
                while i > 0 and pos < n - k:
                    pos += k
                    i -= 1

                # If i == 0, that means after going back T steps, we never
                # left the inherited region, so this final position j at step T
                # is forced to match the character word[pos] in the original.
                if i == 0:
                    # So final_char = word[pos] must equal original_char = word[j].
                    if pos in forced:
                        # It's already forced to match forced[pos], check for conflict
                        if forced[pos] != word[j]:
                            conflict = True
                            break
                    else:
                        # Impose the new constraint
                        forced[pos] = word[j]

                # else (i>0 or pos>=n-k), it comes from appended region, no constraint.

                if conflict:
                    break

            if conflict:
                continue

            # Now also check that all forced[pos] constraints are consistent
            # with word[pos] itself (since "word[pos]" in the original must
            # match the character we forced it to be).
            for pos_in_original, required_char in forced.items():
                if word[pos_in_original] != required_char:
                    conflict = True
                    break

            if not conflict:
                return T

        # In theory we should always find an answer by T <= n,
        # because eventually we can remove the entire string and re-append it.
        # But just in case, fall back:
        return n