import sys

def main():
    n, L, R = map(int, sys.stdin.readline().split())
    a_list = list(map(int, sys.stdin.readline().split()))
    result = [min(max(a, L), R) for a in a_list]
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()