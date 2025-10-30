import sys

def main() -> None:
    # read all integers from stdin
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n = data[0]
    a = data[1:]
    # safety (according to the statement this is always true)
    # but keep it just in case
    if len(a) != n:
        a = a[:n]

    # ----------------------------------------------------------
    # helper that returns the longest length (in elements, i.e. *2)
    #     of a valid subarray whose first element has index `offset`
    #     parity (offset = 0  -> start index even,
    #               offset = 1 -> start index odd)
    #
    # we scan the array as consecutive "blocks" of two elements:
    #   block k   covers indexes  offset + 2*k  and  offset + 2*k + 1
    # a block is
    #   *  valid  if the two elements are equal      -> value = that element
    #   *  invalid otherwise                         -> value = -1  (a barrier)
    #
    # the wanted subarray corresponds to a consecutive segment of
    # blocks that
    #   * are all valid
    #   * have pair-values all different
    #
    # this is the classic “longest subarray with all distinct elements”
    # on the sequence of block-values, where ‘-1’ behaves as a barrier.
    # ----------------------------------------------------------
    def longest(offset: int) -> int:
        left = 0                   # left border (block index) of current window
        best = 0                   # best window length so far   (in blocks)
        last_pos = {}              # last occurrence (block index) of each value
        block_idx = 0              # current block index while scanning
        i = offset                 # current starting index in the original array

        while i + 1 < n:           # there is a complete block
            # determine block value
            if a[i] == a[i + 1]:
                v = a[i]           # valid block
                # duplicate inside window? slide left border
                if v in last_pos and last_pos[v] >= left:
                    left = last_pos[v] + 1
                last_pos[v] = block_idx
                best = max(best, block_idx - left + 1)
            else:
                # invalid block – resets the window
                left = block_idx + 1
                last_pos.clear()

            block_idx += 1
            i += 2                 # next block

        return best * 2            # each block has length 2 in the original array

    ans = max(longest(0), longest(1))
    print(ans)


if __name__ == "__main__":
    main()