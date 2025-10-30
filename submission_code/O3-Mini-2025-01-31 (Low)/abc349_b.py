def main():
    import sys
    from collections import Counter
    
    input_str = sys.stdin.read().strip()
    if not input_str:
        return
    S = input_str.strip()
    
    # Count frequency of each letter
    freq = Counter(S)
    
    # Count how many letters appear a specific number of times
    count_by_freq = {}
    for count in freq.values():
        count_by_freq[count] = count_by_freq.get(count, 0) + 1
    
    # Check for each appearance value (i) within the range 1 to max frequency in S:
    max_count = max(freq.values())
    good = True
    for i in range(1, max_count+1):
        # only consider i that are present in dictionary, but condition must hold:
        num_letters = count_by_freq.get(i, 0)
        if num_letters not in (0, 2):
            good = False
            break
    
    print("Yes" if good else "No")

if __name__ == '__main__':
    main()