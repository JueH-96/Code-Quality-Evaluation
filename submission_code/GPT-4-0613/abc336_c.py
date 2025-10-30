def good_integer(n):
    return int(bin(n)[2:]) * 2

n = int(input().strip())
print(good_integer(n))