def main():
    S = input().strip()
    freq = {}
    for c in S:
        freq[c] = freq.get(c, 0) + 1
    
    max_count = max(freq.values())
    candidates = [ch for ch, cnt in freq.items() if cnt == max_count]
    
    print(min(candidates))

if __name__ == "__main__":
    main()