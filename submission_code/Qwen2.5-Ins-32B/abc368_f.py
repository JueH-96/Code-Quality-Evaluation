import sys

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def solve():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    a = list(map(int, data[1:]))
    
    xor_sum = 0
    for num in a:
        if is_prime(num):
            xor_sum ^= 1
        else:
            count = 0
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    count += 1
                    if num // i != i:
                        count += 1
            xor_sum ^= count
    
    if xor_sum == 0:
        print("Bruno")
    else:
        print("Anna")

solve()