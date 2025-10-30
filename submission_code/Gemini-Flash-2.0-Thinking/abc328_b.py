def solve():
    n = int(input())
    d = list(map(int, input().split()))

    def is_repdigit(m_str, d_str):
        if not m_str or not d_str:
            return False
        first_digit = m_str[0]
        for char in m_str:
            if char != first_digit:
                return False
        for char in d_str:
            if char != first_digit:
                return False
        return True

    repdigit_count = 0
    for i in range(1, n + 1):
        for j in range(1, d[i - 1] + 1):
            month_str = str(i)
            day_str = str(j)
            if is_repdigit(month_str, day_str):
                repdigit_count += 1
    print(repdigit_count)

solve()