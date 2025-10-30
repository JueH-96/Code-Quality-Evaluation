def min_insertions_to_balance(S):
    # Initialize counters for 'i' and 'o'
    count_i = 0
    count_o = 0
    
    # Count the occurrences of 'i' and 'o'
    for char in S:
        if char == 'i':
            count_i += 1
        elif char == 'o':
            count_o += 1
    
    # To satisfy the conditions:
    # 1. The total length must be even
    # 2. The number of 'i's must equal the number of 'o's
    # Calculate the number of insertions needed
    insertions = 0
    
    # If the total length is odd, we need to add one character to make it even
    if (count_i + count_o) % 2 != 0:
        insertions += 1
    
    # The number of 'i's and 'o's must be equal
    # If there are more 'i's, we need to add (count_i - count_o) 'o's
    # If there are more 'o's, we need to add (count_o - count_i) 'i's
    insertions += abs(count_i - count_o)
    
    return insertions

# Read input
import sys
input = sys.stdin.read
S = input().strip()

# Get the result and print it
result = min_insertions_to_balance(S)
print(result)