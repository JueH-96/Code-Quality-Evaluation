from collections import defaultdict

def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    allowed_chars = {'a', 't', 'c', 'o', 'd', 'e', 'r'}

    # Count characters for S and T
    s_count = defaultdict(int)
    t_count = defaultdict(int)
    s_non_allowed = defaultdict(int)
    t_non_allowed = defaultdict(int)
    s_at = 0
    t_at = 0

    for c in S:
        if c == '@':
            s_at += 1
        elif c in allowed_chars:
            s_count[c] += 1
        else:
            s_non_allowed[c] += 1

    for c in T:
        if c == '@':
            t_at += 1
        elif c in allowed_chars:
            t_count[c] += 1
        else:
            t_non_allowed[c] += 1

    # Check non-allowed characters (not @) have the same counts
    valid = True
    for c in s_non_allowed:
        if t_non_allowed.get(c, 0) != s_non_allowed[c]:
            valid = False
            break
    for c in t_non_allowed:
        if s_non_allowed.get(c, 0) != t_non_allowed[c]:
            valid = False
            break
    if not valid:
        print("No")
        return

    # Check sum of allowed characters plus @ counts are equal
    sum_S_allowed = sum(s_count.values())
    sum_T_allowed = sum(t_count.values())
    if (sum_S_allowed + s_at) != (sum_T_allowed + t_at):
        print("No")
        return

    # Calculate sum_R
    sum_R = 0
    for c in allowed_chars:
        diff = s_count[c] - t_count[c]
        if diff > 0:
            sum_R += diff

    if sum_R > t_at:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()