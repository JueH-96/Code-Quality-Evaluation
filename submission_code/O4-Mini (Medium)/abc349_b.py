def main():
    import sys
    from collections import Counter

    S = sys.stdin.readline().strip()
    # Count frequency of each character
    char_counts = Counter(S)
    # Count how many characters have each frequency
    freq_counts = Counter(char_counts.values())

    # For each frequency i that actually occurs,
    # freq_counts[i] must be either 2
    # Frequencies that don't occur implicitly have count 0, which is allowed.
    for freq, count_letters in freq_counts.items():
        if count_letters != 2:
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    main()