def solve():
    n = int(input())
    result = ""
    for i in range(n + 1):
        divisor = -1
        for j in range(1, 10):
            if n % j == 0 and i % (n // j) == 0:
                if divisor == -1:
                    divisor = j
                else:
                    divisor = min(divisor, j)
        if divisor == -1:
            result += "-"
        else:
            result += str(divisor)
    print(result)

solve()