import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    counts = [0] * (n - 1)
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            counts[i] = 1
    prefix = [0] * (n)
    for i in range(n - 1):
        prefix[i + 1] = prefix[i] + counts[i]
    for _ in range(q):
        l, r = map(int, sys.stdin.readline().split())
        if l >= r:
            print(0)
        else:
            print(prefix[r - 1] - prefix[l - 1])

if __name__ == "__main__":
    main()