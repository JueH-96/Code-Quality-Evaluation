import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10000)
    input = sys.stdin.readline

    N = int(input().strip())
    A = list(map(int, input().split()))
    # Precompute sum for every mask
    total_masks = 1 << N
    mask_sum = [0] * total_masks
    for mask in range(1, total_masks):
        # get lowest set bit
        lsb = mask & -mask
        bit_idx = (lsb.bit_length() - 1)
        prev = mask ^ lsb
        mask_sum[mask] = mask_sum[prev] + A[bit_idx]

    full_mask = (1 << N) - 1
    results = set()

    def dfs(unused_mask, curr_xor):
        if unused_mask == 0:
            results.add(curr_xor)
            return
        # pick smallest index in unused_mask
        lsb = unused_mask & -unused_mask
        i = (lsb.bit_length() - 1)
        rem = unused_mask ^ lsb
        # enumerate all subsets of rem
        sub = rem
        while True:
            block = sub | lsb
            # compute new unused
            next_unused = unused_mask ^ block
            dfs(next_unused, curr_xor ^ mask_sum[block])
            if sub == 0:
                break
            sub = (sub - 1) & rem

    dfs(full_mask, 0)
    print(len(results))


if __name__ == "__main__":
    main()