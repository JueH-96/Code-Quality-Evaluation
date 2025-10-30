import sys
from collections import Counter

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    c = Counter(a)
    s = sum(c[i] % 2 for i in c)
    print('Anna' if s % 2 == 0 else 'Bruno')

if __name__ == "__main__":
    main()