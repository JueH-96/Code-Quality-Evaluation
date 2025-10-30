import sys
from collections import Counter

def main():
    s = sys.stdin.read().strip()
    counts = Counter(s)
    freq_counts = Counter(counts.values())
    for v in freq_counts.values():
        if v != 2:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()