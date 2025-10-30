def main():
    import sys
    input = sys.stdin.readline
    from collections import defaultdict

    # We will keep a global dictionary (size -> 0 or 1) indicating how many
    # slimes of each size remain after all possible merges at that size.
    #
    # The key insight is that merging can be done "on the fly" via a carry
    # mechanism: whenever we add x new slimes of size s, they merge (pair up)
    # with whatever is currently in cnt[s]. Each merge of two-size-s slimes
    # becomes one-size-(2*s) slime, so effectively we do:
    #
    #   new_count = cnt[s] + x
    #   pairs     = new_count // 2
    #   leftover  = new_count % 2
    #
    # The "pairs" becomes a "carry" to size 2*s, and we repeat until no more
    # carries can propagate.
    #
    # Eventually, each size in cnt will hold at most 1 slime. The final answer
    # is simply the sum of those remainders in cnt.

    cnt = defaultdict(int)

    def add_slimes(size, number):
        carry = number
        cur = size
        while carry > 0:
            new_count = cnt[cur] + carry
            carry = new_count >> 1  # number of pairs
            leftover = new_count & 1  # 0 or 1
            if leftover == 1:
                cnt[cur] = 1
            else:
                # If leftover is 0 we remove that size from the dictionary
                # to keep memory usage smaller (no slimes left of that size).
                if cur in cnt:
                    del cnt[cur]
            cur <<= 1

    N = int(input())
    for _ in range(N):
        s, c = map(int, input().split())
        add_slimes(s, c)

    # The result is simply the sum of all 0/1 values in cnt
    print(sum(cnt.values()))

# Do not forget to call main()
if __name__ == "__main__":
    main()