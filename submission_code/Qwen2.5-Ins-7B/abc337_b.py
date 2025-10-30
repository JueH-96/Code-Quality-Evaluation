# YOUR CODE HERE
s = input()
a_count = s.count('A')
b_count = s.count('B')
c_count = s.count('C')

if a_count > 0 and b_count > 0 and c_count > 0 and (a_count + b_count + c_count) == len(s) and (a_count == b_count == c_count or (a_count == b_count and (c_count == a_count + b_count)) or (b_count == c_count and (a_count == b_count + c_count)) or (c_count == a_count and (b_count == c_count + a_count))):
    print('Yes')
else:
    print('No')