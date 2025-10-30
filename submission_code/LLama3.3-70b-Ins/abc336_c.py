from itertools import product

def generate_good_integers(n):
    """Generate the nth smallest good integer."""
    # Define the possible digits for good integers
    digits = ['0', '2', '4', '6', '8']
    
    # Initialize a counter for good integers
    count = 0
    
    # Iterate over all possible lengths of good integers
    for length in range(1, 100):
        # Generate all possible good integers of the current length
        for p in product(digits, repeat=length):
            # Convert the generated digits to a string
            num_str = ''.join(p)
            
            # Skip the number if it starts with 0 and has more than one digit
            if num_str[0] == '0' and len(num_str) > 1:
                continue
            
            # Increment the counter
            count += 1
            
            # If the counter reaches n, return the current good integer
            if count == n:
                return int(num_str)

# Read the input from stdin
n = int(input())

# Generate and print the nth smallest good integer
print(generate_good_integers(n))