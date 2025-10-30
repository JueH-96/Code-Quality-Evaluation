import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    L = [0] * N
    R = [0] * N
    for i in range(N):
        l, r = map(int, input().split())
        L[i] = l
        R[i] = r
    L.sort()
    R.sort()
    ptr = 0
    count_non = 0
    # count_non will accumulate the number of non‑intersecting pairs
    for l in L:
        # advance ptr over all processed r_j that end before l
        while ptr < N and R[ptr] < l:
            ptr += 1
        # ptr is now the count of processed r_j < l
        count_non += ptr
    total_pairs = N * (N - 1) // 2
    # intersecting pairs = total_pairs - non_intersecting_pairs
    print(total_pairs - count_non)

main()