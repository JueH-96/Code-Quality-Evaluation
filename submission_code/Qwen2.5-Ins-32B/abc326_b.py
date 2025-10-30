def is_326_like(num):
    hundreds = num // 100
    tens = (num // 10) % 10
    ones = num % 10
    return hundreds * tens == ones

def find_next_326_like(N):
    while not is_326_like(N):
        N += 1
    return N

N = int(input())
print(find_next_326_like(N))