import math

def solve():
    l, r = map(int, input().split())

    def largest_power_of_2_divisor(n):
        if n == 0:
            return -1
        k = 0
        while n % 2 == 0:
            n //= 2
            k += 1
        return k

    def largest_power_of_2_le(n):
        if n <= 0:
            return -1
        return math.floor(math.log2(n))

    current_l = l
    result = []

    while current_l < r:
        p1 = largest_power_of_2_divisor(current_l)
        p2 = largest_power_of_2_le(r - current_l)

        if p1 == -1: # Should not happen if current_l > 0
            power = p2
        elif p2 == -1:
            power = p1
        else:
            power = min(p1, p2)

        length = 2**power
        next_r = current_l + length
        result.append((current_l, next_r))
        current_l = next_r

    print(len(result))
    for start, end in result:
        print(start, end)

solve()