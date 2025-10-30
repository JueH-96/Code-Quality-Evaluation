# YOUR CODE HERE
def is_326_like(number):
    hundreds = number // 100
    tens = (number // 10) % 10
    ones = number % 10
    return hundreds * tens == ones

def find_smallest_326_like(N):
    for num in range(N, 1000):
        if is_326_like(num):
            return num
    return -1  # This should not happen as per constraints

N = int(input())
print(find_smallest_326_like(N))