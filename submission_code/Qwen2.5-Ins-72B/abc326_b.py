# YOUR CODE HERE
def is_326_like(number):
    hundreds = number // 100
    tens = (number // 10) % 10
    ones = number % 10
    return hundreds * tens == ones

def find_smallest_326_like(N):
    while True:
        if is_326_like(N):
            return N
        N += 1

if __name__ == "__main__":
    N = int(input().strip())
    print(find_smallest_326_like(N))