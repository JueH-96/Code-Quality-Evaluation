import sys
from collections import Counter

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        ptr += 1
        a = list(map(int, input[ptr:ptr+n]))
        ptr += n
        count = Counter(a)
        total = 0
        # Add pairs where x == y
        for c in count.values():
            total += c * (c - 1) // 2
        # Add pairs (1, 2)
        total += count.get(1, 0) * count.get(2, 0)
        print(total)

if __name__ == "__main__":
    main()