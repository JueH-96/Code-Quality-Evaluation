import bisect

def main():
    import sys
    data = sys.stdin.read().splitlines()
    first_line = data[0].split()
    N = int(first_line[0])
    T = first_line[1]
    m = len(T)
    strings = data[1:N+1]

    pre_list = []
    suf_list = []

    for s in strings:
        # Compute pre
        current = 0
        for c in s:
            if current < m and c == T[current]:
                current += 1
        pre_list.append(current)

        # Compute suf
        current_suf = m - 1
        count = 0
        for c in reversed(s):
            if current_suf >= 0 and c == T[current_suf]:
                current_suf -= 1
                count += 1
        suf_list.append(count)

    pre_sorted = sorted(pre_list)
    ans = 0

    for j in range(N):
        s = suf_list[j]
        required = max(0, m - s)
        if required <= 0:
            ans += N
        else:
            idx = bisect.bisect_left(pre_sorted, required)
            ans += len(pre_sorted) - idx

    print(ans)

if __name__ == "__main__":
    main()