import sys

def is_ok(s: str, t: str) -> bool:
    """return True if edit-distance between s and t is at most 1"""
    ls, lt = len(s), len(t)
    if abs(ls - lt) > 1:
        return False

    # same length  -> need at most one substitution
    if ls == lt:
        diff = 0
        for i in range(ls):
            if s[i] != t[i]:
                diff += 1
                if diff > 1:
                    return False
        return True

    # make sure s is the longer one (length difference == 1)
    if lt > ls:
        s, t = t, s          # swap so that |s| = |t|+1
        ls, lt = lt, ls

    # now: len(s) = len(t)+1
    i = j = 0
    skipped = False
    while j < lt and i < ls:
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            if skipped:                      # already skipped once
                return False
            skipped = True
            i += 1                           # skip the extra character in the longer string
    return True


def main() -> None:
    data = sys.stdin.buffer.read().splitlines()
    if not data:
        return

    first = data[0].decode().split()
    n = int(first[0])
    t = first[1]

    ok_indices = []
    for idx in range(1, n + 1):
        s = data[idx].decode().strip()
        if is_ok(s, t):
            ok_indices.append(str(idx))

    print(len(ok_indices))
    if ok_indices:
        print(' '.join(ok_indices))


if __name__ == "__main__":
    main()