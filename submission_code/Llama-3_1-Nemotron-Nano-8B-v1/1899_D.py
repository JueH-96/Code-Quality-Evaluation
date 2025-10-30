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
        freq = Counter(a)
        sum_pairs = 0
        for cnt in freq.values():
            sum_pairs += cnt * (cnt - 1) // 2
        count1 = freq.get(1, 0)
        count2 = freq.get(2, 0)
        sum_pairs += count1 * count2
        print(sum_pairs)

if __name__ == "__main__":
    main()