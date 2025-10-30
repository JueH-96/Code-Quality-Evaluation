# YOUR CODE HERE
from itertools import combinations

def generate_321_like_numbers():
    numbers = []
    for r in range(1, 11):  # Digits can be from 1 to 9, so max length is 9
        for comb in combinations(range(1, 10), r):
            num = int(''.join(map(str, sorted(comb, reverse=True))))
            numbers.append(num)
    return sorted(numbers)

def find_kth_321_like_number(K):
    numbers = generate_321_like_numbers()
    return numbers[K-1]

import sys
K = int(sys.stdin.read().strip())
print(find_kth_321_like_number(K))