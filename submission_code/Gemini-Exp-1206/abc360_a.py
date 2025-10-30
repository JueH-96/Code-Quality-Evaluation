s = input()
rice_index = s.find('R')
miso_index = s.find('M')
if rice_index < miso_index:
    print('Yes')
else:
    print('No')