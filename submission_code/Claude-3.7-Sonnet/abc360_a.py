# YOUR CODE HERE
s = input().strip()
rice_pos = s.find('R')
miso_pos = s.find('M')
if rice_pos < miso_pos:
    print("Yes")
else:
    print("No")