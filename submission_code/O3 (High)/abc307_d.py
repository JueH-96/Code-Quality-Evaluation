import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().rstrip()

    # stack items: [index_of_'(', number_of_parentheses_inside_not_removed]
    stack = []
    intervals = []                 # intervals to be removed : (l , r) inclusive

    for i, ch in enumerate(S):
        if ch == '(':
            if stack:
                stack[-1][1] += 1   # one more parenthesis inside current outer '('
            stack.append([i, 0])
        elif ch == ')':
            if not stack:
                continue            # unmatched ')'
            idx_open, inner_cnt = stack.pop()
            if inner_cnt == 0:
                # substring [idx_open, i] has no other parentheses -> deletable
                intervals.append((idx_open, i))
                if stack:
                    stack[-1][1] -= 1   # this pair disappears from its outer '('
        # letters need no special handling

    # build difference array to mark characters to be removed
    diff = [0] * (N + 1)
    for l, r in intervals:
        diff[l] += 1
        diff[r + 1] -= 1

    res = []
    cur = 0
    for i, ch in enumerate(S):
        cur += diff[i]
        if cur == 0:                # not inside any removed interval
            res.append(ch)

    print(''.join(res))

if __name__ == "__main__":
    main()