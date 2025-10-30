import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    INF = float('inf')
    suffix_min = [INF] * (M + 2)  # 1-based to M

    for _ in range(N):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        if R < suffix_min[L]:
            suffix_min[L] = R

    # Compute suffix min
    for x in range(M-1, 0, -1):
        suffix_min[x] = min(suffix_min[x], suffix_min[x+1])

    total = M * (M + 1) // 2
    sum_forbidden = 0

    for l in range(1, M+1):
        current_min = suffix_min[l]
        if current_min <= M:
            sum_forbidden += M - current_min + 1

    print(total - sum_forbidden)

if __name__ == '__main__':
    main()