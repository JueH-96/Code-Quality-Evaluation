# YOUR CODE HERE
n = int(input())
s = input()
t_count = 0
a_count = 0
for i in range(n):
    if s[i] == 'T':
        t_count += 1
    else:
        a_count += 1
    if t_count > a_count and t_count > (n // 2):
        print('T')
        exit()
    if a_count > t_count and a_count > (n // 2):
        print('A')
        exit()
if t_count > a_count:
    print('T')
elif a_count > t_count:
    print('A')
else:
    if t_count == a_count:
        if s.find('T') < s.find('A'):
            print('T')
        else:
            print('A')