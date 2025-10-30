def is_snake_number(n):
    # Convert number to string to process digits
    s = str(n)
    first_digit = int(s[0])
    
    # Check if first digit is strictly larger than all other digits
    for i in range(1, len(s)):
        if int(s[i]) >= first_digit:
            return False
    return True

def count_snake_numbers_in_range(start, end, length, first_digit):
    # If length is 1, we're at the last digit
    if length == 1:
        count = 0
        # Count valid last digits (must be less than first_digit)
        for d in range(10):
            if d < first_digit:
                num = int(str(first_digit) + '0' * (length - 1))
                if start <= num <= end:
                    count += 1
        return count
    
    count = 0
    # For each possible digit at current position
    for d in range(10):
        if d >= first_digit:
            continue
            
        # Calculate the number with current prefix and all zeros
        prefix = str(first_digit) + '0' * (length - 1)
        current_min = int(prefix)
        
        # If this prefix is already beyond our range, break
        if current_min > end:
            break
            
        # Calculate the number with current prefix and all nines
        prefix = str(first_digit) + '9' * (length - 1)
        current_max = int(prefix)
        
        # If this maximum is less than our start, continue
        if current_max < start:
            continue
            
        # If the entire range of numbers with this prefix is within our bounds
        if start <= current_min and current_max <= end:
            count += 10 ** (length - 1)
        else:
            # Recursively count numbers with one less digit
            count += count_snake_numbers_in_range(start, end, length - 1, first_digit)
            
    return count

def solve(L, R):
    # Convert to strings to get lengths
    L_str, R_str = str(L), str(R)
    L_len, R_len = len(L_str), len(R_str)
    
    total = 0
    # For each possible length of numbers
    for length in range(L_len, R_len + 1):
        # For each possible first digit
        for first_digit in range(1, 10):
            # Calculate minimum and maximum numbers of this length
            min_num = 10 ** (length - 1)
            max_num = (10 ** length) - 1
            
            # Adjust range according to L and R
            range_start = max(L, min_num)
            range_end = min(R, max_num)
            
            if range_start > range_end:
                continue
                
            # Add count of snake numbers for this length and first digit
            total += count_snake_numbers_in_range(range_start, range_end, length, first_digit)
            
    return total

# Read input
L, R = map(int, input().split())
# Print output
print(solve(L, R))