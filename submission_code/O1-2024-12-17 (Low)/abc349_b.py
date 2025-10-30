def main():
    import sys

    S = sys.stdin.readline().strip()
    
    # Count the frequency of each letter
    from collections import Counter
    counter = Counter(S)
    
    # Build a dictionary where keys are frequencies and values are counts of how many letters have that frequency
    freq_counts = Counter(counter.values())
    
    # For each frequency from 1 to the maximum frequency in the string:
    max_freq = max(freq_counts.keys())
    for i in range(1, max_freq + 1):
        # How many letters appear exactly i times?
        count_i = freq_counts.get(i, 0)
        # If this count is not 0 or 2, it's not a good string
        if count_i not in (0, 2):
            print("No")
            return
    
    # If we never violated the condition, print "Yes"
    print("Yes")

# Do not forget to call main
main()