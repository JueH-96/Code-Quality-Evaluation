import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    intervals = []
    for _ in range(N):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        intervals.append((L, R))
    # Sort intervals by R ascending, then L descending
    intervals.sort(key=lambda x: (x[1], -x[0]))
    max_L = 0
    forbidden = 0
    for L, R in intervals:
        if L > max_L:
            contribution = (L - max_L) * (M - R + 1)
            forbidden += contribution
            max_L = L
    total_pairs = M * (M + 1) // 2
    ans = total_pairs - forbidden
    print(ans)

if __name__ == '__main__':
    main()