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
        a = list(map(int, input[ptr:ptr + n]))
        ptr += n
        freq = Counter(a)
        equal_pairs = 0
        for f in freq.values():
            equal_pairs += f * (f - 1) // 2
        cnt1 = freq.get(1, 0)
        cnt2 = freq.get(2, 0)
        total = equal_pairs + cnt1 * cnt2
        print(total)

if __name__ == '__main__':
    main()