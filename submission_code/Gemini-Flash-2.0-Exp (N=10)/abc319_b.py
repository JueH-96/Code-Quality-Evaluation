def solve():
    n = int(input())
    result = ""
    for i in range(n + 1):
        smallest_divisor = None
        for j in range(1, 10):
            if n % j == 0:
                if i % (n // j) == 0:
                    if smallest_divisor is None or j < smallest_divisor:
                        smallest_divisor = j
        if smallest_divisor is None:
            result += "-"
        else:
            result += str(smallest_divisor)
    print(result)

solve()