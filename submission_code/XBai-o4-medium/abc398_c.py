import sys
from collections import Counter

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    freq = Counter(a)
    candidates = []
    for i in range(n):
        if freq[a[i]] == 1:
            candidates.append((a[i], i + 1))
    if not candidates:
        print(-1)
    else:
        max_candidate = max(candidates, key=lambda x: x[0])
        print(max_candidate[1])

if __name__ == "__main__":
    main()