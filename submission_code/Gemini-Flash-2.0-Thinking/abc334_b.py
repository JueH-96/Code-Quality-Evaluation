def solve():
    a, m, l, r = map(int, input().split())

    def ceil_div(n, d):
        if d == 0:
            raise ValueError("Cannot divide by zero")
        return (n + d - 1) // d if n >= 0 else n // d

    def floor_div(n, d):
        if d == 0:
            raise ValueError("Cannot divide by zero")
        return n // d if n >= 0 else (n - d + 1) // d

    if m > 0:
        min_k_numerator = l - a
        if min_k_numerator > 0:
            min_k = ceil_div(min_k_numerator, m)
        else:
            min_k = floor_div(min_k_numerator, m)

        max_k_numerator = r - a
        max_k = floor_div(max_k_numerator, m)

        count = max(0, max_k - min_k + 1)
        print(count)
    elif m < 0:
        # Swap min and max logic due to negative M
        min_k_numerator = r - a
        if min_k_numerator > 0:
            min_k = ceil_div(min_k_numerator, m)
        else:
            min_k = floor_div(min_k_numerator, m)

        max_k_numerator = l - a
        max_k = floor_div(max_k_numerator, m)

        count = max(0, max_k - min_k + 1)
        print(count)
    else:
        if l <= a <= r:
            print(1)
        else:
            print(0)

solve()