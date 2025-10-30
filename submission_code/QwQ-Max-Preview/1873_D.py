import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        count = 0
        last = -1
        for i in range(n):
            if s[i] == 'B' and i > last:
                count += 1
                s_start = min(i, n - k)
                last = s_start + k - 1
        print(count)

if __name__ == "__main__":
    main()