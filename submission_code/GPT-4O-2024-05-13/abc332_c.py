# YOUR CODE HERE
def min_tshirts_needed(N, M, S):
    plain_tshirts = M
    logo_tshirts = 0
    max_logo_needed = 0
    
    for i in range(N):
        if S[i] == '1':
            if plain_tshirts > 0:
                plain_tshirts -= 1
            else:
                logo_tshirts += 1
        elif S[i] == '2':
            logo_tshirts += 1
        
        if S[i] == '0':
            plain_tshirts = M
            max_logo_needed = max(max_logo_needed, logo_tshirts)
            logo_tshirts = 0
    
    max_logo_needed = max(max_logo_needed, logo_tshirts)
    
    return max_logo_needed

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
S = data[2]

# Print the result
print(min_tshirts_needed(N, M, S))