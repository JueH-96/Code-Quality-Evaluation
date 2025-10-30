# YOUR CODE HERE
s = input()
r_index = s.find('R')
m_index = s.find('M')
if r_index < m_index:
    print("Yes")
else:
    print("No")