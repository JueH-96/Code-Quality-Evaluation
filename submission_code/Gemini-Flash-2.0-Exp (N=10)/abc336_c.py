def solve():
    n = int(input())
    
    if n == 0:
        print(0)
        return

    digits = []
    while n > 0:
        digits.append(n % 5)
        n //= 5
    
    digits.reverse()
    
    result = 0
    for digit in digits:
        result = result * 10 + digit * 2
    
    print(result)

solve()