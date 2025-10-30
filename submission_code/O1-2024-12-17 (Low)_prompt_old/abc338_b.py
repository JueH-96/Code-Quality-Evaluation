def solve():
    import sys
    
    S = sys.stdin.readline().strip()
    
    # Count frequencies
    freq = {}
    for c in S:
        freq[c] = freq.get(c, 0) + 1
    
    # Find the maximum frequency
    max_freq = max(freq.values())
    
    # Among the characters with max frequency, choose the lexicographically smallest
    candidates = [c for c, count in freq.items() if count == max_freq]
    answer = min(candidates)
    
    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()