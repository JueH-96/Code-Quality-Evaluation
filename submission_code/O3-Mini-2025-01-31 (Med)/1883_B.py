def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    out_lines = []
    
    # For each test case:
    for _ in range(t):
        n = int(data[index])
        k = int(data[index+1])
        index += 2
        s = data[index]
        index += 1

        # Count frequencies of each letter.
        freq = [0] * 26
        for ch in s:
            freq[ord(ch) - 97] += 1
        
        # Count how many characters appear an odd number of times.
        odd_count = sum(1 for count in freq if count % 2)
        
        # After removing k characters, the length of the new string is:
        L = n - k
        # For a string of length L to be rearranged into a palindrome,
        # the number of characters that occur an odd number of times must be:
        #   0 if L is even, or 1 if L is odd.
        allowed = L % 2
        
        # We can choose which characters to remove. The idea is to
        # “fix” the odd counts by removing one occurrence from such characters.
        # However, sometimes we might need to “create” an odd count when the counts are all even
        # but the palindrome (of odd length) requires one odd count.
        # In other words, the minimal removals needed is:
        #    cost = |odd_count - allowed|
        cost = abs(odd_count - allowed)
        
        # Notice that once we have done the minimal removals,
        # any further removals can be done in pairs (removing two characters from any frequency)
        # without disturbing the parity condition required for a palindrome.
        # Hence k - cost must be even.
        if k >= cost and (k - cost) % 2 == 0:
            out_lines.append("YES")
        else:
            out_lines.append("NO")
    
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()