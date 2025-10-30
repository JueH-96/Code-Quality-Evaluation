# YOUR CODE HERE
def count_arithmetic_x(A, B):
    count = 0
    # The possible values for x can be derived from the arithmetic sequence condition
    # x can be calculated as 2A - B or 2B - A
    # We also need to consider the cases where A and B are equal
    possible_x = {2 * A - B, 2 * B - A}
    
    # We can also consider the midpoints
    if (A + B) % 2 == 0:
        possible_x.add((A + B) // 2)
    
    # Count how many unique integers we have
    for x in possible_x:
        if 1 <= x <= 100:
            count += 1
            
    return count

import sys
input = sys.stdin.read
A, B = map(int, input().strip().split())
result = count_arithmetic_x(A, B)
print(result)