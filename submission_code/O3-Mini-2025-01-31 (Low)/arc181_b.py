def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    t = int(data[0].strip())
    line_index = 1
    # An important idea:
    # We are given a fixed word S and two “instructions” X and Y.
    # f(S,T,X) means that for each character in X—if it is '0' we “append” S and if '1' we append an unknown word T.
    # We want to decide whether there exists at least one string T (which may be empty) such that
    #    f(S,T,X) = f(S,T,Y).
    #
    # One first observation is that the total length of these concatenations must be the same.
    # Let |T| = L. Then:
    #    total length for X = (#zeros in X)*|S| + (#ones in X)*L,
    #    total length for Y = (#zeros in Y)*|S| + (#ones in Y)*L.
    # Setting them equal,
    #    (#zeros in X)*|S| + (#ones in X)*L = (#zeros in Y)*|S| + (#ones in Y)*L.
    # Rearranging,
    #    ( (#ones in X) - (#ones in Y) )*L = ( (#zeros in Y) - (#zeros in X) )*|S|.
    #
    # If (#ones in X) ≠ (#ones in Y) then L is forced to be
    #      L = ( (#zeros in Y) - (#zeros in X) ) * |S| / ( (#ones in X) - (#ones in Y) ).
    # This must be a nonnegative integer.
    #
    # If (#ones in X) == (#ones in Y) then automatically
    #    (#zeros in X) == (#zeros in Y) must hold (else no solution exists)
    # and the two instructions X and Y must be “compatible” in the sense that they force the pieces from S to fall in the same positions.
    # In fact, when (#ones in X)==(#ones in Y) and since the number of S segments is equal,
    # a necessary and sufficient condition turns out to be that X and Y are actually exactly equal.
    # (Because each occurrence of '0' attaches S and S is fixed, so the order must match.)
    #
    # Once L is known (or free in the case L is arbitrary) the next step is to “simulate” the two constructions letter‐by‐letter.
    # We traverse (in parallel) the constructed strings f(S,T,X) and f(S,T,Y). For a segment from S we know the letter.
    # For a segment from T, we have an unknown word T but all copies of T must be identical.
    # When one side is “fixed” (coming from S) and the other is from T,
    # we can “set” the appropriate letter of T (if not already determined) and check it for consistency.
    # When both letters are from T (but from potentially different positions within T), no cross‐constraint is imposed.
    #
    # We now implement this idea.
    out_lines = []
    for _ in range(t):
        if line_index >= len(data): 
            break
        S = data[line_index].strip()
        line_index += 1
        X = data[line_index].strip()
        line_index += 1
        Y = data[line_index].strip()
        line_index += 1

        count1_X = X.count('1')
        count1_Y = Y.count('1')
        count0_X = len(X) - count1_X
        count0_Y = len(Y) - count1_Y

        # Case 1: The number of T segments is the same.
        # Then total-length equality forces (count0_X == count0_Y).
        # In that case the segmentation is exactly the same so we must have X and Y identical.
        if count1_X == count1_Y:
            if count0_X != count0_Y or X != Y:
                out_lines.append("No")
            else:
                out_lines.append("Yes")
            continue

        # Case 2: The number of T segments differ.
        # Then L is forced:
        denom = count1_X - count1_Y  # nonzero since count1_X != count1_Y
        num = (count0_Y - count0_X) * len(S)
        # L must be an integer.
        if denom == 0 or num % denom != 0:
            out_lines.append("No")
            continue
        L = num // denom
        if L < 0:
            out_lines.append("No")
            continue

        # We now simulate the letter‐by‐letter matching.
        # For each pattern (X or Y), we keep:
        #   pointer i (or j) for which segment we are in,
        #   and pos which is our position within the current segment.
        # For segments coming from '0', the segment length is len(S) and letters are known.
        # For segments coming from '1', the segment length is L and letters come from T.
        #
        # We maintain an array assignT[0..L-1] (if L > 0) to record already “chosen” letters for T.
        # (If L==0, segments for T mean an empty string; we simulate that by finishing the segment immediately.)
        i, pos_i = 0, 0
        j, pos_j = 0, 0
        if L > 0:
            assignT = [None] * L
        else:
            assignT = []  # when L==0, all T segments are empty.
        possible = True

        # The simulation stops when both patterns are completely processed.
        while i < len(X) or j < len(Y):
            # If one finished before the other, it’s impossible.
            if i == len(X) or j == len(Y):
                possible = False
                break

            # Get the current letter from the left (pattern X)
            if X[i] == '0':
                # We are reading a letter from S.
                if pos_i >= len(S):
                    possible = False
                    break
                c1 = S[pos_i]  # fixed letter
                src1 = 'S'
                seg_len_1 = len(S)
            else:
                # We are reading a letter from T.
                if L == 0:
                    # An empty T segment: simulate it as finished immediately.
                    c1 = None
                    src1 = 'T_empty'
                    seg_len_1 = 0
                else:
                    if pos_i >= L:
                        possible = False
                        break
                    src1 = 'T'
                    c1 = assignT[pos_i]  # might be None
                    seg_len_1 = L

            # Similarly for the right (pattern Y)
            if Y[j] == '0':
                if pos_j >= len(S):
                    possible = False
                    break
                c2 = S[pos_j]
                src2 = 'S'
                seg_len_2 = len(S)
            else:
                if L == 0:
                    c2 = None
                    src2 = 'T_empty'
                    seg_len_2 = 0
                else:
                    if pos_j >= L:
                        possible = False
                        break
                    src2 = 'T'
                    c2 = assignT[pos_j]
                    seg_len_2 = L

            # Now, compare the letters.
            # When both come from fixed S segments, they must equal.
            if src1 == 'S' and src2 == 'S':
                if c1 != c2:
                    possible = False
                    break
            # When one comes from S and the other from T.
            elif src1 == 'S' and src2 == 'T':
                # We “set” the unknown letter from T if not yet set.
                if c2 is None:
                    assignT[pos_j] = c1
                elif c2 != c1:
                    possible = False
                    break
            elif src1 == 'T' and src2 == 'S':
                if c1 is None:
                    assignT[pos_i] = c2
                elif c1 != c2:
                    possible = False
                    break
            # When both are from T segments:
            # They are from copies of T. Notice that they come from possibly different positions in T.
            # Since T is arbitrary the i-th letter and the j-th letter of T might be different.
            # (They are not “coupled” unless one side paired with an S, which forces a value.)
            # So no constraint is imposed in that case.
            # There is a special case when T is empty and we simulate that as T_empty.
            # (If both segments are T_empty, that means that T was chosen to be empty, and the segment gives nothing.)
            # Our simulation here works correctly.

            # Now advance both pointers.
            # For left:
            if X[i] == '0':
                pos_i += 1
                if pos_i == len(S):
                    i += 1
                    pos_i = 0
            else:
                if L == 0:
                    i += 1
                    pos_i = 0
                else:
                    pos_i += 1
                    if pos_i == L:
                        i += 1
                        pos_i = 0

            # For right:
            if Y[j] == '0':
                pos_j += 1
                if pos_j == len(S):
                    j += 1
                    pos_j = 0
            else:
                if L == 0:
                    j += 1
                    pos_j = 0
                else:
                    pos_j += 1
                    if pos_j == L:
                        j += 1
                        pos_j = 0

        out_lines.append("Yes" if possible and i == len(X) and j == len(Y) else "No")
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()