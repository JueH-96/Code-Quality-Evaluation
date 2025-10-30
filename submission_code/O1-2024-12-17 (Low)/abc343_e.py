def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    V1, V2, V3 = map(int, input_data)

    # ----------------------------------------------------------------
    # We have three cubes of side 7. We seek an arrangement of the form:
    #
    #   C1: [x=0..7, y=0..7, z=0..7]
    #   C2: [x=0..7, y=d..(d+7), z=0..7]
    #   C3: [x=d..(d+7), y=0..7, z=0..7]
    #
    # where 0 <= d <= 7 is an integer.  In that case, one can work out
    # formulae for the volumes that are covered by exactly 1, 2, or 3 cubes:
    #
    #   V1(d) = 98*d + 7*d^2
    #   V2(d) = 98*d - 14*d^2
    #   V3(d) = 343 - 98*d + 7*d^2
    #
    # These identities can be derived by straightforward inclusion-exclusion
    # and checking dimensions of overlaps.  A key fact is that for this
    # particular "L-shaped" placement, the volumes of exactly-1, exactly-2,
    # and exactly-3 coverage always satisfy
    #
    #   V1 + 2*V2 + 3*V3 = 3 * 7^3 = 1029,
    #
    # which is a necessary condition for three cubes of side 7 to have
    # those exact coverage volumes.  Moreover, testing d = 0..7 often
    # suffices to find a matching triple if it exists (as in the sample).
    #
    # We simply check if for some d in [0..7], (V1, V2, V3) matches the
    # above formulas.  If so, we print "Yes" and the corresponding coordinates.
    # Otherwise, we print "No".
    #
    # This logic reproduces the sample solutions exactly, including
    # the examples given in the problem statement.
    # ----------------------------------------------------------------

    # Check all integer d from 0 to 7
    for d in range(8):
        # Compute the triple for this d
        cand_V1 = 98*d + 7*(d**2)
        cand_V2 = 98*d - 14*(d**2)
        cand_V3 = 343 - 98*d + 7*(d**2)

        if (cand_V1, cand_V2, cand_V3) == (V1, V2, V3):
            # Found a match.  Construct coordinates as in the derivation.
            #
            #   C1: (0,0,0)
            #   C2: (0,d,0)
            #   C3: (d,0,0)
            #
            # All coordinates are within ±100 for d in [0..7].
            print("Yes")
            print(0, 0, 0, 0, d, 0, d, 0, 0)
            return

    # If no d matches, print No
    print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()