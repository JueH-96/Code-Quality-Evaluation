def max_value_less_equal(S, N):
    from itertools import product

    # Generate all combinations of S by replacing '?' with '0' and '1'
    combinations = product('01', repeat=S.count('?'))
    max_value = -1

    for combo in combinations:
        # Create a new string by replacing '?' with the current combination
        temp_S = S
        for char in combo:
            temp_S = temp_S.replace('?', char, 1)
        
        # Convert the binary string to an integer
        value = int(temp_S, 2)
        
        # Check if the value is less than or equal to N
        if value <= N:
            max_value = max(max_value, value)

    return max_value

import sys
input = sys.stdin.read
data = input().strip().split()
S = data[0]
N = int(data[1])

result = max_value_less_equal(S, N)
print(result)