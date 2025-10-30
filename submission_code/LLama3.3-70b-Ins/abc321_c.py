from itertools import combinations

def generate_321_like_numbers():
    """Generate all possible 321-like numbers."""
    numbers = []
    for d in range(1, 10):  # Maximum number of digits is 9
        for digits in combinations(range(9, 0, -1), d):
            num = int(''.join(map(str, digits)))
            numbers.append(num)
    numbers.sort()
    return numbers

def find_kth_smallest(k):
    """Find the K-th smallest 321-like number."""
    numbers = generate_321_like_numbers()
    return numbers[k - 1]

k = int(input())
result = find_kth_smallest(k)
print(result)