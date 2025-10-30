def main():
    import sys
    
    S = sys.stdin.readline().strip()
    
    # Count frequency of each character
    freq = {}
    for ch in S:
        freq[ch] = freq.get(ch, 0) + 1
    
    # Find the maximum frequency
    max_count = max(freq.values())
    
    # Among characters with max_count, pick the smallest lexicographically
    # Since keys are single letters, we can just iterate in sorted order
    for ch in sorted(freq.keys()):
        if freq[ch] == max_count:
            print(ch)
            return

if __name__ == "__main__":
    main()