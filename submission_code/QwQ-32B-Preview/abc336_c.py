def decimal_to_base5(M):
    if M == 0:
        return [0]
    digits = []
    while M > 0:
        digits.append(M % 5)
        M = M // 5
    digits.reverse()
    return digits

def find_nth_good_integer(N):
    M = N - 1
    base5_digits = decimal_to_base5(M)
    good_digits = [str(d * 2) for d in base5_digits]
    result = ''.join(good_digits)
    print(result)

# Read input
N = int(input())
find_nth_good_integer(N)