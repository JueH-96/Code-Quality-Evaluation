# Read the input from stdin
s = input()

# Check if the plate of rice is to the left of the plate of miso soup
if s.index('R') < s.index('M'):
    print('Yes')
else:
    print('No')