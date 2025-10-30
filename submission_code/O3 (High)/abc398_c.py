import sys
from collections import Counter

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    a = list(map(int, data[1:1 + n]))

    freq = Counter(a)

    best_value = -1          # largest unique value found so far
    best_label = -1          # its corresponding person label (1-indexed)

    for i, val in enumerate(a):
        if freq[val] == 1:   # the value is unique
            if val > best_value:
                best_value = val
                best_label = i + 1   # convert to 1-based label

    print(best_label)

if __name__ == "__main__":
    main()