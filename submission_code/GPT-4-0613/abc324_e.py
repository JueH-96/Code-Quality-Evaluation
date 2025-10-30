import sys
from collections import defaultdict

def subsequence_exists(s, t):
    t = iter(t)
    return all(c in t for c in s)

def main():
    n, t = sys.stdin.readline().split()
    n = int(n)
    s = [sys.stdin.readline().strip() for _ in range(n)]
    count = 0
    for i in range(n):
        for j in range(n):
            if subsequence_exists(t, s[i] + s[j]):
                count += 1
    print(count)

if __name__ == "__main__":
    main()