def solve():
    import sys
    
    S = sys.stdin.readline().strip()
    from collections import Counter
    
    freq_counter = Counter(S)
    freqs = list(freq_counter.values())
    max_freq = max(freqs)
    
    # Count how many letters appear i times for each i
    for i in range(1, max_freq + 1):
        count_i = sum(1 for f in freqs if f == i)
        if count_i not in (0, 2):
            print("No")
            return
    
    print("Yes")

def main():
    solve()

if __name__ == "__main__":
    main()