S = input().strip()
r_pos = S.find('R')
m_pos = S.find('M')
print("Yes" if r_pos < m_pos else "No")