# YOUR CODE HERE
S = input().strip()
r_pos = S.index('R')
m_pos = S.index('M')
print('Yes' if r_pos < m_pos else 'No')