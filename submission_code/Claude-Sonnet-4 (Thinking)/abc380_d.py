S = input().strip()
Q = int(input())
K_list = list(map(int, input().split()))

def flip_char(c):
    return c.upper() if c.islower() else c.lower()

def find_char(pos, operations, s):
    if operations == 0:
        return s[pos - 1]  # pos is 1-indexed
    
    half_length = (2 ** (operations - 1)) * len(s)
    
    if pos <= half_length:
        return find_char(pos, operations - 1, s)
    else:
        char = find_char(pos - half_length, operations - 1, s)
        return flip_char(char)

# Find minimum operations needed
max_K = max(K_list)
operations = 0
length = len(S)
while length < max_K:
    length *= 2
    operations += 1

results = []
for K in K_list:
    results.append(find_char(K, operations, S))

print(' '.join(results))