# YOUR CODE HERE
def find_middle_brother(sab, sac, sbc):
    if sab == '<' and sac == '<':
        if sbc == '<':
            return 'B'
        else:
            return 'C'
    elif sab == '<' and sac == '>':
        return 'A'
    elif sab == '>' and sac == '<':
        return 'A'
    elif sab == '>' and sac == '>':
        if sbc == '<':
            return 'C'
        else:
            return 'B'

sab, sac, sbc = input().split()
print(find_middle_brother(sab, sac, sbc))