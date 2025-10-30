# Read input string
S = input().strip()

# Keep replacing leftmost WA with AC until no WA remains
while "WA" in S:
    # Find leftmost occurrence of WA and replace with AC
    S = S.replace("WA", "AC", 1)

# Print final string
print(S)