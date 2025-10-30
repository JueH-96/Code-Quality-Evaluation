# Read input
s = input().strip()

# Find indices of 'R' and 'M'
r_index = s.find('R')
m_index = s.find('M')

# Check if R is to the left of M
if r_index < m_index:
    print("Yes")
else:
    print("No")