sab, sac, sbc = input().split()

if (sab == '>' and sac == '<') or (sab == '<' and sac == '>'):
    print("A")
elif (sab == '<' and sbc == '<') or (sab == '>' and sbc == '>'):
    print("B")
elif (sac == '<' and sbc == '>') or (sac == '>' and sbc == '<'):
    print("C")