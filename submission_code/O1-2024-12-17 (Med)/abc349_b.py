def main():
    import sys
    S = sys.stdin.readline().strip()
    
    # Count the frequency of each character
    freq = {}
    for ch in S:
        freq[ch] = freq.get(ch, 0) + 1
    
    # Count how many characters appear i times
    freq_counts = {}
    for f in freq.values():
        freq_counts[f] = freq_counts.get(f, 0) + 1
    
    max_freq = max(freq_counts.keys())  # The highest frequency any character has
    
    # Check for each frequency from 1 to max_freq
    for i in range(1, max_freq + 1):
        count_i = freq_counts.get(i, 0)  # Number of characters that appear exactly i times
        if count_i not in (0, 2):
            print("No")
            return
    
    print("Yes")

# Do not forget to call main
main()