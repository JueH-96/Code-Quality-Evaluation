def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    for _ in range(t):
        n, k = map(int, (data[idx], data[idx+1]))
        idx += 2
        s = data[idx]
        idx += 1
        
        # Count frequencies of each character
        freq = [0] * 26
        for ch in s:
            freq[ord(ch) - ord('a')] += 1
        
        # Count how many characters have an odd frequency
        odd_count = sum(f % 2 for f in freq)
        
        # Length of the string after removing exactly k characters
        L = n - k
        
        # If we remove all characters, that (empty string) is trivially a palindrome
        if L == 0:
            print("YES")
            continue
        
        # We want the final arrangement to be a palindrome of length L.
        # Palindromes of even length must have 0 odd frequencies,
        # while palindromes of odd length must have exactly 1 odd frequency.
        # The "cost" here is how many times we need to flip odd frequencies to even
        # (or flip an even frequency to odd if we need exactly one odd but have none).
        
        # Case 1: L is even => final odd_count must be 0
        #   - If odd_count > 0, cost = odd_count
        #   - If odd_count = 0, cost = 0
        #
        # Case 2: L is odd => final odd_count must be 1
        #   - If odd_count > 0, cost = odd_count - 1
        #   - If odd_count = 0, cost = 1  (we flip one even freq to odd by removing 1 character)
        
        if L % 2 == 0:
            cost = odd_count if odd_count > 0 else 0
        else:
            if odd_count > 0:
                cost = odd_count - 1
            else:
                cost = 1
        
        # After paying the "cost" in single removals to fix the number of odd freq,
        # the remainder of our k removals can be spent removing characters in pairs
        # (which does not affect odd/even frequencies).
        leftover = k - cost
        
        # The leftover must be non-negative and even for us to succeed.
        if leftover >= 0 and leftover % 2 == 0:
            print("YES")
        else:
            print("NO")

# Do not forget to call main()
if __name__ == "__main__":
    main()