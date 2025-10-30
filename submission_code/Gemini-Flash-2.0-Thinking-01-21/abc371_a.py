s_ab, s_ac, s_bc = input().split()

if s_ab == '<' and s_ac == '<' and s_bc == '<':
    print("B")
elif s_ab == '<' and s_ac == '<' and s_bc == '>':
    print("C")
elif s_ab == '<' and s_ac == '>' and s_bc == '<':
    print("A")
elif s_ab == '<' and s_ac == '>' and s_bc == '>':
    print("A")
elif s_ab == '>' and s_ac == '<' and s_bc == '<':
    print("A")
elif s_ab == '>' and s_ac == '<' and s_bc == '>':
    print("B")
elif s_ab == '>' and s_ac == '>' and s_bc == '<':
    print("C")
elif s_ab == '>' and s_ac == '>' and s_bc == '>':
    print("B")