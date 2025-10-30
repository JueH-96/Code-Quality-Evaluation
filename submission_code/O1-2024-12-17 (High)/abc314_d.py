def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    S = input().rstrip('
')
    Q = int(input())

    # partialOp[i] will hold (time_of_update, char) for the last update to S[i].
    # If time_of_update == 0, it means "no explicit update; use original S[i]".
    partialOp = [(0, None) for _ in range(N)]

    # Track the last global operation (and its time).
    # 0 means "no global operation yet",
    # 2 means "convert all to lowercase",
    # 3 means "convert all to uppercase".
    lastGlobalTime = 0
    lastGlobalType = 0

    for i in range(1, Q + 1):
        t, x, c = sys.stdin.readline().split()
        t = int(t)
        if t == 1:
            # t=1 => partial update
            x = int(x) - 1
            partialOp[x] = (i, c)
        else:
            # t=2 or t=3 => global operation
            lastGlobalTime = i
            lastGlobalType = t
            # x and c are not used for t=2 or t=3 but are read to match input format

    result = []
    for i in range(N):
        t_p, ch = partialOp[i]
        # If no partial update was set, use original character
        if ch is None:
            ch = S[i]

        # If this position's last update time < the time of the last global operation,
        # then its final form is overshadowed by that global transform.
        if t_p < lastGlobalTime:
            if lastGlobalType == 2:
                ch = ch.lower()
            else:  # lastGlobalType == 3
                ch = ch.upper()

        result.append(ch)

    print("".join(result))

# Don't forget to call main().
main()