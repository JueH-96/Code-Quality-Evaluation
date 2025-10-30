def main():
    import sys
    input_str = sys.stdin.read().strip()  # Read entire input
    if not input_str:
        return
    s = input_str.split()[0]  # In case input includes extra spaces or lines
    
    # Build frequency dictionary
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    # Find maximum frequency
    max_freq = max(freq.values())
    
    # Get characters that have max frequency and return the lexicographically smallest one
    best_char = min(char for char, count in freq.items() if count == max_freq)
    sys.stdout.write(best_char)

if __name__ == "__main__":
    main()