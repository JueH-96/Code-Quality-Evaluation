# YOUR CODE HERE
def nth_good_integer(N):
    if N == 1:
        return 0
    
    N -= 1  # Adjust for 0 being the first good integer
    length = 1
    count = 5  # Number of good integers with 1 digit (2, 4, 6, 8)
    
    while N > count:
        N -= count
        length += 1
        count *= 5
    
    result = [0] * length
    digits = [2, 4, 6, 8]
    
    for i in range(length - 1, -1, -1):
        idx = (N - 1) // (5 ** i)
        result[length - 1 - i] = digits[idx]
        N -= idx * (5 ** i)
    
    return int(''.join(map(str, result)))

N = int(input())
print(nth_good_integer(N))