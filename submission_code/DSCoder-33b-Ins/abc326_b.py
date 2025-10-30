# YOUR CODE HERE

def find_326_like_number(N):
    while True:
        hundreds = N // 100
        tens = (N // 10) % 10
        ones = N % 10
        if hundreds * tens == ones:
            return N
        N += 1

N = int(input())
print(find_326_like_number(N))