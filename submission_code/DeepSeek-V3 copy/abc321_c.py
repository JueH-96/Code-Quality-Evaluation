from itertools import combinations

def find_kth_321_like_number(K):
    result = []
    # Generate all possible lengths of the number
    for length in range(1, 11):  # Since 9876543210 is the largest 10-digit 321-like number
        # Generate all combinations of digits for the current length
        for digits in combinations(range(9, -1, -1), length):
            # Ensure the digits are strictly decreasing
            if all(digits[i] > digits[i+1] for i in range(len(digits)-1)):
                # Convert the tuple of digits to a number
                num = int(''.join(map(str, digits)))
                result.append(num)
                # If we have enough numbers, return the K-th one
                if len(result) >= K:
                    return result[K-1]
    return result[K-1]

# Read input
K = int(input())
# Find and print the K-th 321-like number
print(find_kth_321_like_number(K))