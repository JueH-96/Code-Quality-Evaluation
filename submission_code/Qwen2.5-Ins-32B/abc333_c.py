def generate_repunit(length):
    return int('1' * length)

def find_nth_sum_of_three_repunits(n):
    # Initialize the lengths of the three repunits
    a, b, c = 1, 1, 1
    count = 0
    
    while True:
        # Generate the repunits
        rep_a = generate_repunit(a)
        rep_b = generate_repunit(b)
        rep_c = generate_repunit(c)
        
        # Calculate the sum of the three repunits
        sum_repunits = rep_a + rep_b + rep_c
        
        # Increment the count
        count += 1
        
        if count == n:
            return sum_repunits
        
        # Increment the lengths of the repunits
        if a <= b and a <= c:
            a += 1
        elif b <= a and b <= c:
            b += 1
        else:
            c += 1

# Read input from stdin
n = int(input())

# Find the N-th smallest integer that can be expressed as the sum of exactly three repunits
result = find_nth_sum_of_three_repunits(n)

# Write the answer to stdout
print(result)