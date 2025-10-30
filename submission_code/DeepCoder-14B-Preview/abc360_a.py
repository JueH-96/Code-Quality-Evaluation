# Read the input string
s = input().strip()

# Find the indices of 'R' and 'M'
r_pos = s.index('R')
m_pos = s.index('M')

# Check if R is to the left of M
if r_pos < m_pos:
    print("Yes")
else:
    print("No")