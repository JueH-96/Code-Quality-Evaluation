# Read the input
s1, s2 = input().split()

# Determine the correct oyster based on the conditions
if s1 == 'sick' and s2 == 'sick':
    print(1)
elif s1 == 'sick' and s2 == 'fine':
    print(2)
elif s1 == 'fine' and s2 == 'sick':
    print(3)
else:  # both fine, so it's 4
    print(4)