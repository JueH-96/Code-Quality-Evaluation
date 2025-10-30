import sys

S = input()
T = input()

if len(S) != len(T):
    print("No")
    sys.exit()

s_list = list(S)
t_list = list(T)

# Replace all '@' with a valid character
for i in range(len(s_list)):
    if s_list[i] == '@':
        s_list[i] = 'a'
    if t_list[i] == '@':
        t_list[i] = 'a'

# Sort the characters in each row
s_list.sort()
t_list.sort()

# Check if the two rows are the same
if ''.join(s_list) == ''.join(t_list):
    print("Yes")
else:
    print("No")