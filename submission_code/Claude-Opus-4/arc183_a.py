from math import factorial

def count_sequences(remaining):
    """Count the number of sequences that can be formed with the given remaining counts"""
    total = sum(remaining)
    if total == 0:
        return 1
    
    result = factorial(total)
    for count in remaining:
        result //= factorial(count)
    return result

def find_kth_sequence(n, k, target_index):
    """Find the target_index-th sequence (1-indexed) in lexicographical order"""
    remaining = [k] * n  # remaining[i] = number of times we still need to use (i+1)
    result = []
    
    while sum(remaining) > 0:
        # Try each number from 1 to n
        for num in range(n):
            if remaining[num] == 0:
                continue
            
            # Temporarily use this number
            remaining[num] -= 1
            
            # Count sequences starting with current prefix + this number
            count = count_sequences(remaining)
            
            if target_index <= count:
                # This is the correct number to place
                result.append(num + 1)
                break
            else:
                # Skip these sequences
                target_index -= count
                remaining[num] += 1
    
    return result

# Read input
n, k = map(int, input().split())

# Calculate total number of sequences
total_sequences = factorial(n * k)
for _ in range(n):
    total_sequences //= factorial(k)

# Find the median sequence (1-indexed)
target_index = (total_sequences + 1) // 2

# Get the sequence
sequence = find_kth_sequence(n, k, target_index)

# Print the result
print(' '.join(map(str, sequence)))