def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    pos = 1
    
    for _ in range(t):
        n = int(data[pos]); pos += 1
        k = int(data[pos]); pos += 1
        s = data[pos]; pos += 1
        
        # Count frequency of each character
        freq = [0]*26
        for ch in s:
            freq[ord(ch) - ord('a')] += 1
        
        # Number of characters that appear with odd frequency
        num_odd = sum(f % 2 for f in freq)
        # Number of distinct characters in s
        distinct_count = sum(f > 0 for f in freq)
        
        # We want to remove exactly k characters, so the remaining length is M
        M = n - k
        # A palindrome of length M can have at most M%2 characters of odd frequency
        final_odds = M % 2
        
        # Depending on whether we have more or fewer odd frequencies than needed,
        # we either reduce or increase the count of odd frequencies by "toggling."
        # Toggling odd->even or even->odd costs exactly 1 removal, and the rest of
        # the removals must be done in pairs (which do not affect parity).
        
        if final_odds <= num_odd:
            # Need to reduce odd frequencies from num_odd to final_odds
            needed_toggles = num_odd - final_odds
            # We must at least do needed_toggles single removals (one from each
            # of those odd frequencies), and the remaining k - needed_toggles must
            # be an even number to be done in pairs.
            if k >= needed_toggles and (k - needed_toggles) % 2 == 0:
                print("YES")
            else:
                print("NO")
        else:
            # Need to increase odd frequencies from num_odd to final_odds
            needed_toggles = final_odds - num_odd
            # We must toggle that many even frequencies to become odd. So we need
            # enough even-freq characters and the leftover k - needed_toggles must be even.
            if needed_toggles <= (distinct_count - num_odd) and k >= needed_toggles and (k - needed_toggles) % 2 == 0:
                print("YES")
            else:
                print("NO")

# Do not forget to call main!
if __name__ == "__main__":
    main()