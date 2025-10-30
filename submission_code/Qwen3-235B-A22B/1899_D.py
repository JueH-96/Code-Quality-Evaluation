import sys
from collections import Counter

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline().strip())
        a = list(map(int, sys.stdin.readline().strip().split()))
        cnt = Counter(a)
        res = 0
        for count in cnt.values():
            res += count * (count - 1) // 2
        res += cnt.get(1, 0) * cnt.get(2, 0)
        print(res)

if __name__ == "__main__":
    main()