import math

def factor(n):
    factors = {}
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n = n // 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n = n // i
        i += 2
    if n > 1:
        factors[n] = 1
    return factors

def generate_square_free_divisors(factors_dict):
    primes = list(factors_dict.keys())
    divisors = [1]
    for p in primes:
        new_divisors = []
        for d in divisors:
            new_divisors.append(d)
            new_divisors.append(d * p)
        divisors = list(set(new_divisors))
    divisors.sort()
    return divisors

def is_valid_number(s):
    return all(c in '123456789' for c in s)

def get_largest_valid_number(m):
    if m == 0:
        return 0
    s = list(str(m))
    n = len(s)
    i = 0
    while i < n:
        if s[i] == '0':
            j = i - 1
            while j >= 0 and s[j] == '0':
                j -= 1
            if j < 0:
                return int('9' * (n - 1)) if n > 1 else 0
            s[j] = str(int(s[j]) - 1)
            if s[j] == '0' and j == 0:
                for k in range(j + 1, n):
                    s[k] = '9'
                if s[j] == '0':
                    return int('9' * (n - 1))
            for k in range(j + 1, n):
                s[k] = '9'
            i = 0
        else:
            i += 1
    num = int(''.join(s))
    if all(c in '123456789' for c in str(num)):
        return num
    else:
        return get_largest_valid_number(num - 1)

def get_previous_valid_number(num):
    return get_largest_valid_number(num - 1)

def find_split(m):
    if m == 1:
        return "1"
    max_num = get_largest_valid_number(m)
    while max_num > 0:
        if max_num > m:
            max_num = get_previous_valid_number(max_num)
            continue
        if m % max_num == 0:
            remainder = m // max_num
            split_remainder = find_split(remainder)
            if split_remainder is not None:
                if split_remainder == "1":
                    return str(max_num)
                else:
                    return f"{max_num}*{split_remainder}"
        max_num = get_previous_valid_number(max_num)
    return None

def main():
    N = int(input().strip())
    if N == 1:
        print(1)
        return
    factors = factor(N)
    square_free_divisors = generate_square_free_divisors(factors)
    for K in square_free_divisors:
        if N % K != 0:
            continue
        x = N // K
        sqrt_x = math.isqrt(x)
        if sqrt_x * sqrt_x != x:
            continue
        M = sqrt_x
        k_str = str(K)
        if not is_valid_number(k_str):
            continue
        split_m = find_split(M)
        if split_m is None:
            continue
        left_part = split_m
        left_tokens = left_part.split('*')
        reversed_tokens = left_tokens[::-1]
        reversed_left_part = '*'.join(reversed_tokens)
        full_string = left_part
        if K != 1:
            full_string += '*' + k_str
        full_string += '*' + reversed_left_part
        if full_string and full_string[0] in '123456789' and full_string == full_string[::-1]:
            print(full_string)
            return
    print(-1)

if __name__ == '__main__':
    main()