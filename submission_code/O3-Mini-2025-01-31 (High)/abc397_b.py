def main():
    import sys
    
    S = sys.stdin.read().strip().split()[0]
    
    # We want to obtain an alternating string T (with even length)
    # of the form "i o i o ..." (1-indexed positions: T[1] = 'i', T[2] = 'o', T[3] = 'i', etc.)
    # such that S appears as a subsequence of T.
    #
    # In other words, we want to insert as few characters as possible into S
    # so that the final string becomes a perfect alternating sequence.
    #
    # A neat observation is that we can “align” S with an infinite alternating sequence:
    #    P[1]=i, P[2]=o, P[3]=i, P[4]=o, ... 
    # We then want to find positions p1, p2, ..., pN (with 1 ≤ p1 < p2 < ... < pN)
    # such that for each k, S[k] = P[p_k]. The final length of T will be at least pN,
    # but it must be even. Therefore, if pN is odd we will need to append one extra character.
    #
    # Our goal is to minimize the final T length (say, M) so that the number of inserted characters,
    # which equals M - len(S), is minimized.
    #
    # Because the pattern P is completely regular (odd positions are 'i' and even positions are 'o'),
    # for each character in S we can compute the smallest possible index in P that can match it,
    # given that the previous matched index was p. (Here we treat p=0 initially meaning no characters matched yet.)
    #
    # Concretely, when matching a character c in S:
    #   - If c == 'i', then it must appear at an odd-numbered position.
    #       If the current index p is even then p+1 is odd, so we choose p+1.
    #       If p is odd then p+1 is even and we must skip one extra to get an odd index, i.e. p+2.
    #   - Similarly, if c == 'o' then it must appear at an even-numbered position.
    #       If p is odd then p+1 is even, else if p is even then we pick p+2.
    
    p = 0
    for c in S:
        if c == 'i':
            # need next odd number > p
            if p % 2 == 0:
                p += 1
            else:
                p += 2
        elif c == 'o':
            # need next even number > p
            if p % 2 == 1:
                p += 1
            else:
                p += 2
        # Since S contains only 'i' and 'o', no else branch is necessary.
    
    # The final T must have even length. If p (the last matched index) is odd, we need to append one extra character.
    if p % 2 == 1:
        p += 1
        
    # The number of inserted characters equals the final length (p) minus the number of original characters.
    answer = p - len(S)
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()