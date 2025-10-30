def solve():
    n = int(input())
    
    digits = [0, 2, 4, 6, 8]
    
    result = ""
    n -= 1
    
    while n >= 0:
        result = str(digits[n % 5]) + result
        n = n // 5 - 1
    
    print(result)

solve()