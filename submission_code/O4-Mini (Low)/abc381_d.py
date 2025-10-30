import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))
    ans_blocks = 0

    # We consider two parities for the starting index of the 2-element blocks: 0 or 1 (0-based)
    for p in (0, 1):
        # Build compressed block array V: each entry is the value if A[i]==A[i+1], else -1
        # i runs over p, p+2, p+4, ... up to N-2
        M = (N - p) // 2  # number of possible blocks of this parity
        V = [0] * M
        for k in range(M):
            i = p + 2 * k
            if i + 1 < N and A[i] == A[i+1]:
                V[k] = A[i]
            else:
                V[k] = -1

        last_pos = {}
        l = 0
        # two-pointer over V to find longest subarray with no -1 and all distinct
        for r in range(M):
            val = V[r]
            if val == -1:
                # invalid block, reset window
                last_pos.clear()
                l = r + 1
            else:
                if val in last_pos and last_pos[val] >= l:
                    # duplicate value in window, advance l
                    l = last_pos[val] + 1
                last_pos[val] = r
                # window length in blocks is r - l + 1
                ans_blocks = max(ans_blocks, r - l + 1)

    # each block is of size 2
    print(ans_blocks * 2)

if __name__ == "__main__":
    main()