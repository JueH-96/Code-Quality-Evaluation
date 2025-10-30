# YOUR CODE HERE
def find_next_326_like_number(N):
    for num in range(N, 1000):
        hundreds = num // 100
        tens = (num // 10) % 10
        ones = num % 10
        if hundreds * tens == ones:
            return num

N = int(input())
print(find_next_326_like_number(N))