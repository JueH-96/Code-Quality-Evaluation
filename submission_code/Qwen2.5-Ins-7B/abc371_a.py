# YOUR CODE HERE
a_b = input()
a_c = input()
b_c = input()

if a_b == '<' and a_c == '<' and b_c == '<':
    print('C')
elif a_b == '>' and a_c == '>' and b_c == '>':
    print('A')
elif a_b == '<' and a_c == '>' and b_c == '<':
    print('B')
elif a_b == '>' and a_c == '<' and b_c == '>':
    print('B')
elif a_b == '<' and a_c == '<' and b_c == '>':
    print('A')
elif a_b == '>' and a_c == '>' and b_c == '<':
    print('C')
elif a_b == '<' and a_c == '>' and b_c == '>':
    print('C')
elif a_b == '>' and a_c == '<' and b_c == '<':
    print('A')