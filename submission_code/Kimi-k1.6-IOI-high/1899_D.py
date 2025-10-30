import sys
from collections import Counter

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        freq = Counter(a)
        res = 0
        for c in freq.values():
            res += c * (c - 1) // 2
        cnt1 = freq.get(1, 0)
        cnt2 = freq.get(2, 0)
        res += cnt1 * cnt2
        print(res)

if __name__ == "__main__":
    main()