import math

def count_square_permutations():
    N = int(input().strip())
    S = input().strip()
    sorted_S = sorted(S)
    max_x = 10**N - 1
    sqrt_max = math.isqrt(max_x)
    count = 0

    for i in range(0, sqrt_max + 1):
        x = i * i
        str_x = str(x)
        if len(str_x) < N:
            str_x = '0' * (N - len(str_x)) + str_x
        if sorted(str_x) == sorted_S:
            count += 1

    print(count)

count_square_permutations()