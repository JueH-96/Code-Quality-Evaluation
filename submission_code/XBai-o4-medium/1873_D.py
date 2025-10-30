import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        covered = 0
        count = 0
        for i in range(n):
            if i >= covered:
                if s[i] == 'B':
                    count += 1
                    covered = i + k
        print(count)

if __name__ == "__main__":
    main()