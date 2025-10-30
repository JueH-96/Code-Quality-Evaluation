def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        k = int(data[index+1])
        index += 2
        s = data[index]
        index += 1

        # Count frequency of each character in s.
        freq = {}
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1

        # Count the number of characters with odd frequency.
        odd_count = sum(1 for cnt in freq.values() if cnt % 2 == 1)

        # After removing k characters, the remaining string length is:
        remaining = n - k
        # A rearranged palindrome for a string of length 'remaining' must have:
        # - 0 odd-count characters if remaining is even.
        # - 1 odd-count character if remaining is odd.
        allowed_odds = remaining % 2

        # We can fix one odd frequency by removing one occurrence from that letter.
        # Hence, if we have more than allowed_odds odd frequencies,
        # the minimum number of removals needed to get a valid set (by fixing parity)
        # is: needed = odd_count - allowed_odds.
        needed = odd_count - allowed_odds if odd_count > allowed_odds else 0

        # After removing exactly 'needed' characters, the parity condition is fixed.
        # The remaining removals (k - needed) have to be done in pairs so they do not disturb
        # the parity balance. Therefore, (k - needed) must be even.
        if k >= needed and (k - needed) % 2 == 0:
            results.append("YES")
        else:
            results.append("NO")
    
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()