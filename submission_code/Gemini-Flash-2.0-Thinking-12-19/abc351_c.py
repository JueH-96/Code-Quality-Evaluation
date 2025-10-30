def solve():
    n = int(input())
    a = list(map(int, input().split()))
    exponents = []
    for exponent in a:
        exponents.append(exponent)
        while len(exponents) >= 2 and exponents[-1] == exponents[-2]:
            last_exponent = exponents.pop()
            exponents.pop()
            exponents.append(last_exponent + 1)
    print(len(exponents))

if __name__ == '__main__':
    solve()