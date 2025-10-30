def is_326_like(num):
    # Extract each digit
    hundreds = num // 100         # Hundreds digit
    tens = (num // 10) % 10       # Tens digit
    ones = num % 10               # Ones digit
    
    # Check if the product of hundreds and tens equals ones
    return hundreds * tens == ones

def find_smallest_326_like(N):
    # Check numbers starting from N until we find a valid one
    for num in range(N, 1000):  # 999 is the largest 3-digit number
        if is_326_like(num):
            return num
    return None  # Should not reach here given the constraints

# Read input
N = int(input())

# Print output
print(find_smallest_326_like(N))