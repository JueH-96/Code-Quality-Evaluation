def solve():
    n = int(input())
    start_hundreds_digit = n // 100
    for hundreds_digit in range(start_hundreds_digit, 10):
        if hundreds_digit == 10:
            break
        for tens_digit in range(10):
            ones_digit = hundreds_digit * tens_digit
            if 0 <= ones_digit <= 9:
                like_number = hundreds_digit * 100 + tens_digit * 10 + ones_digit
                if like_number >= n:
                    print(like_number)
                    return

solve()