# Read the input as a string
s = input()

# Extract the three digits (as characters)
a = s[0] # hundreds digit
b = s[1] # tens digit
c = s[2] # ones digit

# Form the first new string by arranging b, c, a
s1 = b + c + a

# Form the second new string by arranging c, a, b
s2 = c + a + b

# Convert the new strings to integers
int1 = int(s1)
int2 = int(s2)

# Print the two integers separated by a space
print(int1, int2)