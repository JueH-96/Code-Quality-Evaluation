import sys

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    positions = [[] for _ in range(n + 1)]
    for idx in range(1, 3 * n + 1):
        num = A[idx - 1]
        positions[num].append(idx)
    middle = []
    for i in range(1, n + 1):
        middle.append((positions[i][1], i))
    middle.sort()
    print(' '.join(map(str, [x[1] for x in middle])))

if __name__ == "__main__":
    main()