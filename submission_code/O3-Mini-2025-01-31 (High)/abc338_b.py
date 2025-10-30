def main():
    import sys
    S = sys.stdin.read().strip()
    # Initialize a frequency list for each character a-z.
    frequency = [0] * 26
    for ch in S:
        frequency[ord(ch) - ord('a')] += 1
    
    # Determine the maximum frequency.
    max_freq = max(frequency)
    # Find the lexicographically smallest character with the maximum frequency.
    for i in range(26):
        if frequency[i] == max_freq:
            print(chr(i + ord('a')))
            break

if __name__ == '__main__':
    main()