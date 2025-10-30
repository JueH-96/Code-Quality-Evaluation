# YOUR CODE HERE
def generate_321_like_numbers():
    for i in range(1, 10):  # Single digit numbers
        yield i
    
    def generate_decreasing(prefix, max_digit):
        if max_digit == 0:
            yield int(prefix)
        else:
            for d in range(max_digit, 0, -1):
                yield from generate_decreasing(prefix + str(d), d - 1)
    
    for length in range(2, 10):  # Multi-digit numbers
        for start in range(9, length - 1, -1):
            yield from generate_decreasing(str(start), start - 1)

def find_kth_321_like_number(k):
    generator = generate_321_like_numbers()
    for _ in range(k - 1):
        next(generator)
    return next(generator)

k = int(input())
result = find_kth_321_like_number(k)
print(result)