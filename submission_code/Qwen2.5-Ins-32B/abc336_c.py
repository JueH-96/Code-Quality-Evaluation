def find_nth_good_integer(N):
    digits = []
    base = 5
    while N > 0:
        digits.append(N % base)
        N //= base
    digits = digits[::-1]
    even_digits = ['0', '2', '4', '6', '8']
    result = ''.join(even_digits[d] for d in digits)
    return result.lstrip('0') or '0'

if __name__ == "__main__":
    N = int(input().strip())
    print(find_nth_good_integer(N))