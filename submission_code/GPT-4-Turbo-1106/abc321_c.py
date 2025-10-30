from itertools import combinations

def find_kth_321_like_number(k):
    # Generate all possible 321-like numbers with digits 1-9
    digits = '9876543210'
    all_321_like_numbers = []
    for i in range(1, 11):
        all_321_like_numbers.extend([''.join(c) for c in combinations(digits, i)])
    
    # Convert to integers and sort
    all_321_like_numbers = sorted(int(num) for num in all_321_like_numbers)
    
    # Return the k-th number
    return all_321_like_numbers[k-1]

# Read from stdin
k = int(input().strip())

# Write to stdout
print(find_kth_321_like_number(k))