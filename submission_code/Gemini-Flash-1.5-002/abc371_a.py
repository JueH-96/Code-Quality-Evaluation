# YOUR CODE HERE
sab, sac, sbc = input().split()

if sab == '<' and sac == '<' and sbc == '<':
    print('B')
elif sab == '<' and sac == '<' and sbc == '>':
    print('C')
elif sab == '<' and sac == '>' and sbc == '<':
    print('A')
elif sab == '<' and sac == '>' and sbc == '>':
    print('B')
elif sab == '>' and sac == '<' and sbc == '<':
    print('A')
elif sab == '>' and sac == '<' and sbc == '>':
    print('C')
elif sab == '>' and sac == '>' and sbc == '<':
    print('B')
elif sab == '>' and sac == '>' and sbc == '>':
    print('A')