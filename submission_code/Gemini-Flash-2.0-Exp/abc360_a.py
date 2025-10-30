S = input()
r_index = S.find('R')
m_index = S.find('M')

if r_index < m_index:
  print("Yes")
else:
  print("No")