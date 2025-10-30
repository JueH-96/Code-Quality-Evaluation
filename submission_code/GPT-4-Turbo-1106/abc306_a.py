# Read input
N = int(input().strip())
S = input().strip()

# Process and output
result = ''.join([c*2 for c in S])
print(result)