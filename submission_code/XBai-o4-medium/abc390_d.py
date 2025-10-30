import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    max_mask = 1 << N
    sum_for_mask = [0] * max_mask

    for mask in range(max_mask):
        s = 0
        for i in range(N):
            if mask & (1 << i):
                s += A[i]
        sum_for_mask[mask] = s

    dp = [set() for _ in range(max_mask)]
    dp[0].add(0)

    for mask in range(1, max_mask):
        first = (mask & -mask).bit_length() - 1
        mask_without_first = mask ^ (1 << first)
        for sub in range(mask_without_first + 1):
            if (sub & mask_without_first) == sub:
                s = (1 << first) | sub
                rem = mask ^ s
                for x in dp[rem]:
                    dp[mask].add(x ^ sum_for_mask[s])
    
    full_mask = (1 << N) - 1
    print(len(dp[full_mask]))

if __name__ == '__main__':
    main()