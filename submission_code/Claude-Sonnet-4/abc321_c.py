K = int(input())

numbers = []

# Generate all possible 321-like numbers
# For each non-empty subset of digits {0,1,2,3,4,5,6,7,8,9}
for mask in range(1, 1024):  # 2^10 - 1, excluding empty set
    digits = []
    for i in range(10):
        if mask & (1 << i):
            digits.append(i)
    
    # Sort in decreasing order
    digits.sort(reverse=True)
    
    # Convert to number
    num_str = ''.join(map(str, digits))
    numbers.append(int(num_str))

numbers.sort()
print(numbers[K-1])