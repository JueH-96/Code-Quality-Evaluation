import sys

def can_make_equal_with_one_edit(s: str, t: str) -> bool:
    """
    Return True if s can be converted to t using at most one
    insertion, deletion or replacement; otherwise, return False.
    """
    # Already the same
    if s == t:
        return True

    n, m = len(s), len(t)

    # If length difference exceeds 1, need at least |n-m| edits
    if abs(n - m) > 1:
        return False

    # ------------------------------------------------------------------
    # Case 1 : same length  -> possible single REPLACEMENT
    # ------------------------------------------------------------------
    if n == m:
        diff_cnt = 0
        for ch_s, ch_t in zip(s, t):
            if ch_s != ch_t:
                diff_cnt += 1
                if diff_cnt > 1:
                    return False
        # diff_cnt is 1 here (because s != t handled above)
        return True

    # ------------------------------------------------------------------
    # Case 2 : length differs by exactly 1
    #          -> single INSERTION into shorter / DELETION from longer
    # ------------------------------------------------------------------
    # Ensure `shorter` is really the shorter one
    if n > m:
        s, t = t, s        # swap so that s is shorter, t is longer
        n, m = m, n        # (m = n+1 after swap)

    i = j = 0       # indices for shorter (s) and longer (t)
    diff_found = False

    while i < n and j < m:
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            if diff_found:          # already consumed our single allowance
                return False
            diff_found = True       # treat t[j] as the extra char
            j += 1                  # advance only in longer string

    # Any remaining character in `t` (the longer one) is the single extra char
    return True


def main() -> None:
    input_data = sys.stdin.readline
    K = int(input_data().strip())   # K is guaranteed to be 1 in this sub-problem
    S = input_data().strip()
    T = input_data().strip()

    if K == 0:                      # (not needed for this task, but complete)
        print("Yes" if S == T else "No")
        return

    # K == 1 for this task
    print("Yes" if can_make_equal_with_one_edit(S, T) else "No")


if __name__ == "__main__":
    main()