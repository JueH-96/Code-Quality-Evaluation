def decimal_to_base5(n):
    if n == 0:
        return [0]
    digits = []
    while n > 0:
        digits.append(n % 5)
        n = n // 5
    return digits[::-1]

def get_good_integer(N):
    if N == 1:
        return '0'
    M = N - 1
    base5_digits = decimal_to_base5(M)
    mapping = {0: '0', 1: '2', 2: '4', 3: '6', 4: '8'}
    good_digits = [mapping[d] for d in base5_digits]
    return ''.join(good_digits)

def main():
    N = int(input())
    print(get_good_integer(N))

if __name__ == "__main__":
    main()