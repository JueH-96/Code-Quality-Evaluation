from collections import defaultdict
import sys

def main():
    S = sys.stdin.read().strip()
    freq = defaultdict(int)
    for char in S:
        freq[char] += 1
    max_freq = max(freq.values())
    for c in sorted(freq):
        if freq[c] == max_freq:
            print(c)
            break

if __name__ == "__main__":
    main()