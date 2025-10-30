import sys
from collections import defaultdict

def main():
    Q = int(sys.stdin.readline())
    count = defaultdict(int)
    unique = 0
    for _ in range(Q):
        parts = sys.stdin.readline().split()
        if not parts:
            continue
        if parts[0] == '1':
            x = int(parts[1])
            if count[x] == 0:
                unique += 1
            count[x] += 1
        elif parts[0] == '2':
            x = int(parts[1])
            count[x] -= 1
            if count[x] == 0:
                unique -= 1
        elif parts[0] == '3':
            print(unique)

if __name__ == "__main__":
    main()