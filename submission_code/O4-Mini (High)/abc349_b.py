def main():
    import sys
    from collections import Counter

    S = sys.stdin.readline().strip()
    # Count frequency of each letter
    freq = Counter(S)
    # Count how many letters have each frequency
    freq_of_freq = Counter(freq.values())
    # For each frequency i>=1 that occurs,
    # the number of letters with that frequency must be exactly 2
    for count in freq_of_freq.values():
        if count != 2:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()