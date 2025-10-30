n, d = map(int, input().split())
s = input().strip()

# Collect all positions of '@' in the original string
pos_list = [i for i, c in enumerate(s) if c == '@']
remaining = len(pos_list) - d  # Number of '@' remaining after D days

# Initialize the result with all '.' and then set the remaining '@' positions
result = ['.'] * n
for pos in pos_list[:remaining]:
    result[pos] = '@'

print(''.join(result))