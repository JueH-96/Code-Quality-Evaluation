def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    S = data[0]
    n = len(S)
    res = []
    i = 0
    # We will simulate the net effect of the following transformation:
    #  "WA"  →  "AC"
    # performed repeatedly on S.
    #
    # One may show that if there is any occurrence of a letter 'W' immediately
    # followed by an 'A' then eventually (after several steps) all consecutive W’s
    # that immediately precede that A in S will get “converted” as one block:
    # Suppose there is a block W...W (say k many Ws) immediately followed by an A.
    # Then the final effect is to turn that substring into:
    #
    #  "A" + "C"*k
    #
    # For example,
    #  • "WA"   (k=1) becomes "AC"
    #  • "WWA"  (k=2) becomes "ACC"
    #  • "WWWA" (k=3) becomes "ACCC"
    #
    # On any portion where a run of W’s is not immediately followed by an A,
    # nothing happens.
    #
    # Thus, we can scan the string S from left to right. For each occurrence of a
    # letter that is not 'W', we output it as is. When we see a 'W' we measure how
    # many consecutive Ws there are. If the run is directly followed by an A then
    # we output the converted block ("A" then as many "C" as there are Ws) and skip
    # that A. Otherwise we output the Ws unchanged.
    
    while i < n:
        if S[i] == 'W':
            j = i
            while j < n and S[j] == 'W':
                j += 1
            # if there is an 'A' immediately following this run of Ws
            if j < n and S[j] == 'A':
                countW = j - i
                res.append("A" + ("C" * countW))
                i = j + 1  # we skip over the 'A' used for the conversion
            else:
                # no conversion is triggered; output the Ws themselves.
                res.append(S[i:j])
                i = j
        else:
            res.append(S[i])
            i += 1

    sys.stdout.write("".join(res))


if __name__ == '__main__':
    main()