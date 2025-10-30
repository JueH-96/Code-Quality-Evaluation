import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    beans = defaultdict(list)
    for _ in range(n):
        a, c = map(int, sys.stdin.readline().split())
        beans[c].append(a)
    min_deliciousness = []
    for c in beans:
        beans[c].sort()
        min_deliciousness.append(beans[c][0])
    print(max(min_deliciousness))

if __name__ == "__main__":
    main()