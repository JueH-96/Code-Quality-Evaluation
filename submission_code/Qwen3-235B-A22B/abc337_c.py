import sys

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    next_node = [None] * (n + 2)  # Using indices 1 to N
    start = -1
    for j in range(1, n + 1):
        a_j = A[j - 1]
        if a_j == -1:
            start = j
        else:
            next_node[a_j] = j
    res = []
    current = start
    for _ in range(n):
        res.append(str(current))
        current = next_node[current]
    print(' '.join(res))

if __name__ == "__main__":
    main()