def generate_321_like_numbers():
    from itertools import combinations
    
    # Generate all combinations of digits from 1 to 9
    numbers = []
    for length in range(1, 10):  # lengths from 1 to 9
        for combo in combinations(range(1, 10), length):
            # Create a number from the combination in decreasing order
            number = int(''.join(map(str, sorted(combo, reverse=True))))
            numbers.append(number)
    
    # Sort the numbers to get them in increasing order
    numbers.sort()
    return numbers

def find_kth_321_like_number(k):
    numbers = generate_321_like_numbers()
    return numbers[k - 1]  # k is 1-based index

import sys
input = sys.stdin.read

k = int(input().strip())
result = find_kth_321_like_number(k)
print(result)