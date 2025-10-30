import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    a = set(map(int, sys.stdin.readline().split()))
    missing = []
    for i in range(1, n+1):
        if i not in a:
            missing.append(i)
    print(len(missing))
    print(' '.join(map(str, missing)))

if __name__ == "__main__":
    main()