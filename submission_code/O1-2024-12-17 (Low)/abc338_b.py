def main():
    import sys
    
    S = sys.stdin.readline().strip()
    freq = {}
    for ch in S:
        freq[ch] = freq.get(ch, 0) + 1
    
    max_freq = max(freq.values())
    # Filter characters that have max frequency
    candidates = [ch for ch, count in freq.items() if count == max_freq]
    print(min(candidates))

# Do not remove the call to main()
main()