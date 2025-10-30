def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    pos = 1
    # We'll collect results and output them at once.
    results = []
    for _ in range(t):
        n = int(data[pos]); k = int(data[pos+1]); pos += 2
        s = data[pos]; pos += 1

        # Count frequencies and the count of letters with odd frequency.
        odd_count = 0
        freq = {}
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
        for count in freq.values():
            if count % 2 == 1:
                odd_count += 1

        # After removing exactly k characters, the remaining length is:
        L = n - k

        # For a multiset to be rearranged into a palindrome,
        # if L is even every letter must occur an even number of times.
        # If L is odd, exactly one letter must occur an odd number of times.
        #
        # When we remove characters, we can “fix” wrong parities by removing a single character from any letter.
        # However any extra removal must come in pairs (so the parity does not flip again).
        #
        # Thus, think in two parts:
        #   1. Minimal removals required to adjust the multiset to the needed parity:
        #      - If L is even: we must "fix" every odd count 
        #        => minimal required removals = odd_count.
        #      - If L is odd: we must have exactly one odd count.
        #        If there is already at least one letter with an odd count, we can “fix” (odd_count - 1) of them.
        #        Otherwise (which practically cannot happen since L odd forces at least one odd),
        #        we would need to remove one extra character.
        #
        #   2. Then the extra removals beyond the minimal must be performed in pairs.
        
        if L % 2 == 0:
            required = odd_count
        else:
            if odd_count > 0:
                required = odd_count - 1
            else:
                required = 1  # (This case should not occur because an odd total can't have all counts even)

        extra = k - required
        
        # Check two conditions:
        #   (i) We must have k >= minimal required (i.e. extra >= 0)
        #  (ii) The extra removals must be an even number.
        if extra >= 0 and extra % 2 == 0:
            results.append("YES")
        else:
            results.append("NO")
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()