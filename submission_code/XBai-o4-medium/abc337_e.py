import sys

def main():
    N = int(sys.stdin.readline())
    current = 1
    m = 0
    while current < N:
        current *= 2
        m += 1
    print(m)
    tests = []
    for i in range(1, m+1):
        bit_pos = m - i
        bottles = []
        for x in range(1, N+1):
            if ((x-1) >> bit_pos) & 1:
                bottles.append(x)
        tests.append(bottles)
    for t in tests:
        print(len(t), end=' ')
        print(' '.join(map(str, t)))
    S = sys.stdin.readline().strip()
    x_minus_1 = int(S, 2)
    print(x_minus_1 + 1)

if __name__ == "__main__":
    main()