import sys

def main():
    q = int(sys.stdin.readline())
    counts = {}
    for _ in range(q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            x = int(parts[1])
            counts[x] = counts.get(x, 0) + 1
        elif parts[0] == '2':
            x = int(parts[1])
            counts[x] -= 1
            if counts[x] == 0:
                del counts[x]
        elif parts[0] == '3':
            print(len(counts))

if __name__ == "__main__":
    main()