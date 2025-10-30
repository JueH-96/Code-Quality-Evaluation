from collections import Counter
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        ptr += 1
        a = list(map(int, input[ptr:ptr + n]))
        ptr += n
        freq = Counter(a)
        total = 0
        # Add all pairs where a_i = a_j
        for cnt in freq.values():
            total += cnt * (cnt - 1) // 2
        # Add pairs where a_i is 1 and a_j is 2 or vice versa
        if 1 in freq and 2 in freq:
            total += freq[1] * freq[2]
        print(total)

if __name__ == "__main__":
    main()