def minimum_logo_tshirts_needed(N, M, S):
    plain_available = M
    logo_available = 0
    max_logo_needed = 0
    
    for day in S:
        if day == '1':
            if plain_available > 0:
                plain_available -= 1
            else:
                logo_available -= 1
        elif day == '2':
            if logo_available > 0:
                logo_available -= 1
            else:
                max_logo_needed += 1
                logo_available += 1
                logo_available -= 1
        elif day == '0':
            plain_available = M
            logo_available = max_logo_needed
    
    return max_logo_needed

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
M = int(data[1])
S = data[2]

# Calculate and print the result
result = minimum_logo_tshirts_needed(N, M, S)
print(result)