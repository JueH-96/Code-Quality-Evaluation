def main():
    import sys
    from collections import Counter

    S = sys.stdin.readline().strip()
    # Count frequency of each letter
    letter_counts = Counter(S)
    # Count how many letters have each frequency
    freq_counts = Counter(letter_counts.values())
    # Check the condition: for each frequency that occurs,
    # the number of letters with that frequency must be exactly 2.
    for cnt in freq_counts.values():
        if cnt != 2:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()