import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    positions = [[] for _ in range(N + 1)]
    for idx in range(len(A)):
        num = A[idx]
        positions[num].append(idx + 1)
    res = []
    for i in range(1, N + 1):
        mid = positions[i][1]
        res.append((mid, i))
    res.sort()
    print(' '.join(str(x[1]) for x in res))

if __name__ == '__main__':
    main()