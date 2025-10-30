s1, s2 = input().split()
s1_sick = (s1 == 'sick')
s2_sick = (s2 == 'sick')

for x in [1, 2, 3, 4]:
    in_t = x in {1, 2}
    in_a = x in {1, 3}
    if in_t == s1_sick and in_a == s2_sick:
        print(x)
        break