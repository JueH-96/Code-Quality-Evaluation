import sys
from collections import defaultdict

def main():
    n, m = map(int, sys.stdin.readline().split())
    friends = defaultdict(list)
    for _ in range(m):
        a, b = map(int, sys.stdin.readline().split())
        friends[a].append(b)
        friends[b].append(a)
    max_friends = max(len(friends[i]) for i in range(1, n+1))
    print(max_friends + (n - 2*max_friends if n - 2*max_friends > 0 else 0))

if __name__ == "__main__":
    main()