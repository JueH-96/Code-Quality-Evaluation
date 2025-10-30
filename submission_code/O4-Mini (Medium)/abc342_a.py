def main():
    S = input().strip()
    # Count frequencies of each character
    freq = {}
    for ch in S:
        freq[ch] = freq.get(ch, 0) + 1
    # Find the character that appears exactly once
    for i, ch in enumerate(S):
        if freq[ch] == 1:
            # Output is 1-based index
            print(i + 1)
            return

if __name__ == "__main__":
    main()