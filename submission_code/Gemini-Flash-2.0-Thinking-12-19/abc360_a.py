s = input()
index_r = s.find('R')
index_m = s.find('M')

if index_r < index_m:
    print("Yes")
else:
    print("No")