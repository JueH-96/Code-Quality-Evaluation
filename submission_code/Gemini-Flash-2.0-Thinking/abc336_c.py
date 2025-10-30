def solve():
    n = int(input())
    n -= 1

    if n == -1:
        return

    if n == 0:
        print(0)
        return

    good_digits_map = {0: '0', 1: '2', 2: '4', 3: '6', 4: '8'}
    result_digits = []

    if n >= 0:
        while True:
            remainder = n % 5
            result_digits.append(good_digits_map[remainder])
            n //= 5
            if n == 0:
                break

    print("".join(reversed(result_digits)))

solve()