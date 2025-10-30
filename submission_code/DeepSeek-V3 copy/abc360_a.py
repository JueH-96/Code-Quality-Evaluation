# Read the input string
S = input().strip()

# Find the indices of 'R' and 'M'
r_index = S.find('R')
m_index = S.find('M')

# Determine if 'R' is to the left of 'M'
if r_index < m_index:
    print("Yes")
else:
    print("No")