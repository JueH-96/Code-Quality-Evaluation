import sys

# Read the input string from stdin
s = sys.stdin.readline().strip()

# Find the index of 'R' (rice)
r_index = s.index('R')

# Find the index of 'M' (miso soup)
m_index = s.index('M')

# Check if the index of 'R' is less than the index of 'M'
if r_index < m_index:
    print("Yes")
else:
    print("No")