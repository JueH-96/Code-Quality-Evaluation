import sys
from collections import defaultdict

def main():
    H, W = map(int, sys.stdin.readline().split())
    count = defaultdict(int)
    for _ in range(H):
        s = sys.stdin.readline().strip()
        mask = 0
        for c in s:
            mask = (mask << 1) | (1 if c == '1' else 0)
        count[mask] += 1

    n = 1 << W
    A = [0] * n
    for mask in count:
        A[mask] = count[mask]

    B = [0] * n
    for mask in range(n):
        pop = bin(mask).count('1')
        B[mask] = min(pop, W - pop)

    def walsh_hadamard(a, invert):
        n = len(a)
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(h):
                    x = a[i + j]
                    y = a[i + j + h]
                    a[i + j] = x + y
                    a[i + j + h] = x - y
            h <<= 1
        if invert:
            divisor = n
            for i in range(n):
                a[i] = a[i] // divisor

    walsh_hadamard(A, invert=False)
    walsh_hadamard(B, invert=False)

    C = [A[i] * B[i] for i in range(n)]

    walsh_hadamard(C, invert=True)

    min_sum = min(C)
    print(min_sum)

if __name__ == "__main__":
    main()