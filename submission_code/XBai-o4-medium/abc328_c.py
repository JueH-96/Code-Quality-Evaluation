import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    prefix = [0] * n  # prefix[0] is 0, and up to prefix[n-1]

    for i in range(1, n):
        if s[i-1] == s[i]:
            prefix[i] = prefix[i-1] + 1
        else:
            prefix[i] = prefix[i-1]
    
    for _ in range(q):
        l, r = map(int, sys.stdin.readline().split())
        res = prefix[r-1] - prefix[l-1]
        print(res)

if __name__ == "__main__":
    main()