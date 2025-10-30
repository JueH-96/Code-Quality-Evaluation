def main():
    import sys
    input_str = sys.stdin.read().strip()
    if not input_str:
        return
    W, B = map(int, input_str.split())
    tot = W + B
    # Our infinite string S is the infinite repetition of
    # pattern "wbwbwwbwbwbw". Its length is 12, and within one period
    # there are 7 occurrences of "w" and 5 occurrences of "b".
    pattern = "wbwbwwbwbwbw"
    period = len(pattern)
    
    # Instead of processing every possible substring of S (which is infinite),
    # we can use the periodicity and just check all substrings starting at positions
    # with distinct offsets (i.e. 0 through period-1). For each such starting offset,
    # the substring S[i : i+tot] (when taking S as an infinite cycle) is "representative".
    #
    # To avoid doing any modular arithmetic in our check, we can build an extended string
    # that is long enough that any substring starting in the first period and of length tot
    # is contained in it. A safe length is period + tot.
    
    # Calculate how many copies we need:
    reps = (tot // period) + 2  # +2 is safe enough
    extended = pattern * reps
    
    possible = False
    # We only need to check substrings starting at positions 0 to period-1 (distinct cyclic shifts)
    for start in range(period):
        # Get the substring of length tot starting at 'start'
        sub = extended[start:start + tot]
        if sub.count('w') == W and sub.count('b') == B:
            possible = True
            break
    print("Yes" if possible else "No")

if __name__ == '__main__':
    main()