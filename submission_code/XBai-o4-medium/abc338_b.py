import sys
from collections import Counter

def main():
    s = sys.stdin.readline().strip()
    counts = Counter(s)
    max_freq = max(counts.values())
    candidates = [char for char, freq in counts.items() if freq == max_freq]
    print(min(candidates))

if __name__ == "__main__":
    main()