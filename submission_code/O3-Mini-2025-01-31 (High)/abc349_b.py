def main():
    import sys
    from collections import Counter
    
    # Read the input string and strip any trailing whitespace/newlines.
    S = sys.stdin.read().strip()
    
    # Count the frequency of each letter in S.
    letter_counts = Counter(S)
    
    # Count how many letters have a given frequency.
    # freq_counts[i] will be the number of letters that appear exactly i times.
    freq_counts = Counter(letter_counts.values())
    
    # Check for all positive integers i:
    # if any frequency occurs (nonzero count), it must occur for exactly 2 letters.
    for freq, count in freq_counts.items():
        if count != 2:
            print("No")
            return
    
    # If all frequencies that appeared satisfied the condition, then S is a good string.
    print("Yes")

# Call the main function as specified.
if __name__ == "__main__":
    main()