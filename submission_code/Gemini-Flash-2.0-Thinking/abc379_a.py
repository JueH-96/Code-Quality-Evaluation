N = int(input())
s = str(N)
a = int(s[0])
b = int(s[1])
c = int(s[2])

first_num = int(str(b) + str(c) + str(a))
second_num = int(str(c) + str(a) + str(b))

print(first_num, second_num)