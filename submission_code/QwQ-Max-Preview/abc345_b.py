x = int(input())
quotient = x // 10
remainder = x % 10
if remainder != 0:
    quotient += 1
print(quotient)