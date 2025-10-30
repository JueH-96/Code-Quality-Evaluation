import sys
from collections import Counter

def main():
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        n = int(input[idx])
        idx += 1
        a = list(map(int, input[idx:idx + n]))
        idx += n
        cnt = Counter(a)
        total = 0
        for c in cnt.values():
            total += c * (c - 1) // 2
        count_1 = cnt.get(1, 0)
        count_2 = cnt.get(2, 0)
        total += count_1 * count_2
        print(total)

if __name__ == "__main__":
    main()