# Read the input
S = input().strip()

# Find the positions of 'R' and 'M'
r_pos = S.index('R')
m_pos = S.index('M')

# Check if R is to the left of M
if r_pos < m_pos:
    print("Yes")
else:
    print("No")