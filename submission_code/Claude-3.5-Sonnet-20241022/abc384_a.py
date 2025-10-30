# Read input
N, c1, c2 = input().split()
N = int(N)
S = input()

# Process each character: keep c1, replace others with c2
result = ''
for c in S:
    if c == c1:
        result += c1
    else:
        result += c2

# Print result
print(result)