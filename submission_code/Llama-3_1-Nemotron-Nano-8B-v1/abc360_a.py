s = input().strip()
r_pos = s.index('R')
m_pos = s.index('M')
print("Yes" if r_pos < m_pos else "No")