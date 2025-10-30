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
        a = list(map(int, input[idx:idx+n]))
        idx += n
        cnt = Counter(a)
        total = 0
        for v in cnt.values():
            total += v * (v - 1) // 2
        c1 = cnt.get(1, 0)
        c2 = cnt.get(2, 0)
        total += c1 * c2
        print(total)

if __name__ == "__main__":
    main()