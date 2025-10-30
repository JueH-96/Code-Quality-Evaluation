import sys
from collections import Counter

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    a.sort()
    counter = Counter(a)
    keys = list(counter.keys())
    values = list(counter.values())
    acc = [0] * (len(keys) + 1)
    for i in range(len(keys) - 1, -1, -1):
        acc[i] = acc[i + 1] + keys[i] * values[i]
    ans = 0
    for i in range(len(keys)):
        ans += keys[i] * values[i] * (values[i] - 1) // 2
        if i + 1 < len(keys):
            ans += values[i] * (acc[i + 1] - keys[i + 1] * values[i + 1])
    print(ans)

if __name__ == "__main__":
    main()