# Read the input string
S = input().strip()

# Sort the characters and check if they form 'ABC'
sorted_S = ''.join(sorted(S))
if sorted_S == 'ABC':
    print("Yes")
else:
    print("No")