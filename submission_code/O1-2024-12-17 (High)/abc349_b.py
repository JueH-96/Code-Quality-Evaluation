def main():
    import sys
    from collections import Counter
    
    S = sys.stdin.readline().strip()
    letter_counts = Counter(S)
    freq_counts = Counter(letter_counts.values())
    
    max_freq = max(letter_counts.values())  # highest frequency among letters
    
    # Check for each i from 1 up to max_freq if exactly 0 or 2 letters have count i
    for i in range(1, max_freq + 1):
        if freq_counts[i] not in (0, 2):
            print("No")
            return
    
    print("Yes")


# Do not remove the next line
main()