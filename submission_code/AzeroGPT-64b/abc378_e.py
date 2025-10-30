from collections import Counter

def compute_remainder_sums(length, modulus, numbers):
    """
    Calculate the sum of all possible subarray sums modulo a given modulus.
    
    Args:
    length (int): The number of elements in the array.
    modulus (int): The modulus for subarray sums.
    numbers (list): The array of non-negative integers.
    
    Returns:
    int: The sum of all subarray sums modulo the given modulus.
    """
    cumulative_sums, total, current = Counter([0]), 0, 0
    for index, number in enumerate(numbers):
        current = (current + number) % modulus
        total += current * (index + 1) - current * cumulative_sums[current]
        cumulative_sums[current] += 1
    return total

# Reading input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Calculating and printing the result
print(compute_remainder_sums(N, M, A))