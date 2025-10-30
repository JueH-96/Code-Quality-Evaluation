s = input().strip()
r_index = s.find('R')
m_index = s.find('M')
print("Yes" if r_index < m_index else "No")