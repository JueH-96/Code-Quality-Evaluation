# Read the input from stdin
s1, s2 = input().split()

# Determine the type of oyster that causes stomach trouble
if s1 == 'sick' and s2 == 'fine':
    print(2)
elif s1 == 'fine' and s2 == 'sick':
    print(3)
else:
    print(4)