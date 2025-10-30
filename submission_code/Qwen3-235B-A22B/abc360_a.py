# Read the input string
s = input().strip()

# Find the indices of 'R' and 'M'
r_index = s.find('R')
m_index = s.find('M')

# Check if R is to the left of M and print the result
print("Yes" if r_index < m_index else "No")