import sys
from collections import defaultdict

def main():
    q = int(sys.stdin.readline())
    freq = defaultdict(int)
    count = 0
    output = []
    for _ in range(q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            x = int(parts[1])
            if freq[x] == 0:
                count += 1
            freq[x] += 1
        elif parts[0] == '2':
            x = int(parts[1])
            freq[x] -= 1
            if freq[x] == 0:
                count -= 1
        elif parts[0] == '3':
            output.append(str(count))
    print('
'.join(output))

if __name__ == "__main__":
    main()