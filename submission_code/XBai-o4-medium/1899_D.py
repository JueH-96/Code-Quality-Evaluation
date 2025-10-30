import sys
from collections import Counter

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
        cnt = Counter(a)
        same = 0
        for v in cnt.values():
            same += v * (v - 1) // 2
        cross = cnt.get(1, 0) * cnt.get(2, 0)
        print(same + cross)

if __name__ == "__main__":
    main()