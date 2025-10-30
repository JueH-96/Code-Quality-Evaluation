def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast IO helpers:
    # input_data is a list of tokens (strings).
    # We'll parse them as needed.
    # Format:
    # N Q
    # then Q queries:
    #   1 x c
    # or
    #   2 c

    # Parse N, Q
    N = int(input_data[0])
    Q = int(input_data[1])

    # We will maintain:
    # 1) A list "sections" of (start, end, color), kept sorted by "start".
    #    No two consecutive intervals in "sections" will ever share the same color,
    #    and they will not overlap nor be out of order.
    # 2) color_size[c] = how many cells are currently of color c.
    #
    # Initially, cell i is painted color i (1 ≤ i ≤ N).
    # So we have intervals (1,1,1), (2,2,2), ..., (N,N,N).
    # and color_size[i] = 1 for i in [1..N], and 0 otherwise.

    sections = [(i, i, i) for i in range(1, N + 1)]
    color_size = [0] * (N + 1)
    for i in range(1, N + 1):
        color_size[i] = 1

    import bisect

    # A small helper to get length of an interval
    def interval_len(st, ed):
        return ed - st + 1

    # Find the index i in sections such that sections[i] contains x
    # We'll do a binary search by "start".
    def find_interval(x):
        # We want the largest i where sections[i].start <= x
        # We'll use bisect_right on (x, large, large) then subtract 1
        lo = 0
        hi = len(sections)
        while lo < hi:
            mid = (lo + hi) // 2
            if sections[mid][0] > x:
                hi = mid
            else:
                lo = mid + 1
        i = lo - 1
        return i

    # Expand the region of color old_color that contains the interval at index i
    # We'll return the indices [L, R] that define the block of consecutive intervals
    # that all share color old_color and are contiguous (end of one == start of next - 1).
    def expand_region(i, old_color):
        L = i
        R = i
        # expand left
        while L > 0:
            s_prev, e_prev, c_prev = sections[L - 1]
            s_cur, e_cur, c_cur = sections[L]
            if c_prev == old_color and e_prev + 1 == s_cur:
                L -= 1
            else:
                break
        # expand right
        while R < len(sections) - 1:
            s_cur, e_cur, c_cur = sections[R]
            s_next, e_next, c_next = sections[R + 1]
            if c_next == old_color and e_cur + 1 == s_next:
                R += 1
            else:
                break
        return (L, R)

    import sys
    out = []
    idx = 2  # we have consumed input_data[0..1], so the next token is input_data[2]

    for _ in range(Q):
        t = int(input_data[idx]); idx += 1

        if t == 1:
            # query "1 x c"
            x = int(input_data[idx]); idx += 1
            c = int(input_data[idx]); idx += 1
            # Recolor the connected component of x to color c.

            # 1) find the interval containing x
            i = find_interval(x)
            st, ed, old_color = sections[i]
            if old_color == c:
                # No change needed
                continue

            # 2) find the entire contiguous block [L, R] of old_color that contains i
            L, R = expand_region(i, old_color)

            # 3) compute the size of that block
            block_start = sections[L][0]
            block_end   = sections[R][1]
            block_size  = 0
            for j in range(L, R + 1):
                s_j, e_j, col_j = sections[j]
                block_size += interval_len(s_j, e_j)
            # Remove that block from old_color
            color_size[old_color] -= block_size

            # 4) remove those intervals from "sections"
            del sections[L:R+1]

            # We'll create a new interval that initially is exactly that recolored block
            new_start = block_start
            new_end   = block_end
            new_size  = block_size
            new_color = c

            # 5) possibly merge with left neighbor if same color c
            left_nb = L - 1
            if left_nb >= 0:
                s_left, e_left, c_left = sections[left_nb]
                if c_left == new_color and e_left + 1 == new_start:
                    # remove the left neighbor from sections, adjust new_start/new_size
                    size_left = interval_len(s_left, e_left)
                    color_size[new_color] -= size_left  # remove to avoid double-count
                    new_start = s_left
                    new_size += size_left
                    # remove that interval
                    del sections[left_nb]
                    L = left_nb

            # 6) possibly merge with right neighbor if same color c
            #    after removal above, our insertion index is L
            if L < len(sections):
                s_right, e_right, c_right = sections[L]
                if c_right == new_color and new_end + 1 == s_right:
                    size_right = interval_len(s_right, e_right)
                    color_size[new_color] -= size_right
                    new_end = e_right
                    new_size += size_right
                    del sections[L]

            # 7) insert the new merged interval
            sections.insert(L, (new_start, new_end, new_color))
            # 8) add the final size to color_size[new_color]
            color_size[new_color] += new_size

        else:
            # query "2 c"
            c = int(input_data[idx]); idx += 1
            out.append(str(color_size[c]))

    # Print all answers for queries of type 2
    print("
".join(out))

# Do not forget to call main
if __name__ == "__main__":
    main()