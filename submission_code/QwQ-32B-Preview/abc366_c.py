import sys
from collections import defaultdict

def main():
    Q_line = sys.stdin.readline()
    Q = int(Q_line)
    bag = defaultdict(int)
    for _ in range(Q):
        query_line = sys.stdin.readline().strip()
        if query_line.startswith('1'):
            parts = query_line.split()
            if len(parts) != 2:
                continue
            x = int(parts[1])
            bag[x] += 1
        elif query_line.startswith('2'):
            parts = query_line.split()
            if len(parts) != 2:
                continue
            x = int(parts[1])
            bag[x] -= 1
            if bag[x] == 0:
                del bag[x]
        elif query_line == '3':
            print(len(bag))

if __name__ == "__main__":
    main()