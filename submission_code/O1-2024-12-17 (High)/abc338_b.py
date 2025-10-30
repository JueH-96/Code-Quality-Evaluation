def main():
    import sys
    s = sys.stdin.readline().strip()
    
    # Count the frequency of each character
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    
    # Find the maximum frequency
    max_freq = max(freq.values())
    
    # Collect all characters that have the maximum frequency
    candidates = [char for char, count in freq.items() if count == max_freq]
    
    # Among the characters with max frequency, print the earliest alphabetically
    print(min(candidates))

# Do not remove the call to main()
main()