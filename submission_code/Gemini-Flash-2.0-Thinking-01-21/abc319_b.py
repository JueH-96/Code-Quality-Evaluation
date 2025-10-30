def solve():
    n = int(input())
    result_chars = []
    for i in range(n + 1):
        s_i = '-'
        found_divisor = False
        min_divisor = None
        for j in range(1, 10):
            if n % j == 0:
                if i % (n // j) == 0:
                    if min_divisor is None or j < min_divisor:
                        min_divisor = j
                        found_divisor = True
        if found_divisor:
            s_i = str(min_divisor)
        result_chars.append(s_i)
    print("".join(result_chars))

if __name__ == '__main__':
    solve()