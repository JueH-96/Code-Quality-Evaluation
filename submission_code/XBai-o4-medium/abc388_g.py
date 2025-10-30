import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(Q):
        L = int(input[ptr])
        R = int(input[ptr+1])
        ptr += 2
        L0 = L - 1
        R0 = R - 1
        k = R0 - L0 + 1
        m = k // 2
        mid = L0 + m - 1
        i = L0
        j = mid + 1
        count = 0
        while i <= mid and j <= R0:
            if A[j] >= 2 * A[i]:
                count += 1
                i += 1
                j += 1
            else:
                j += 1
        results.append(str(count))
    print('
'.join(results))

if __name__ == '__main__':
    main()