def get_proper_divisors(n):
    divisors = []
    for i in range(1, n):
        if n % i == 0:
            divisors.append(i)
    return divisors

def calculate_mex(values):
    seen_values = set(values)
    i = 0
    while i in seen_values:
        i += 1
    return i

grundy_values = {}
grundy_values[1] = 0

def calculate_grundy(n):
    if n in grundy_values:
        return grundy_values[n]
    proper_divisors = get_proper_divisors(n)
    if not proper_divisors:
        grundy_values[n] = 0
        return 0
    next_grundy_values = []
    for d in proper_divisors:
        next_grundy_values.append(calculate_grundy(d))
    mex_value = calculate_mex(next_grundy_values)
    grundy_values[n] = mex_value
    return mex_value

if __name__ == '__main__':
    n = int(input())
    a = list(map(int, input().split()))
    nim_sum = 0
    for val in a:
        nim_sum ^= calculate_grundy(val)
    if nim_sum != 0:
        print("Anna")
    else:
        print("Bruno")