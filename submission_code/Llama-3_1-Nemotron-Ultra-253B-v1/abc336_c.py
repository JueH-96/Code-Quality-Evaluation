N = int(input())
M = N - 1
if M == 0:
    print(0)
else:
    digits = []
    while M > 0:
        digits.append(M % 5)
        M = M // 5
    digits = digits[::-1]
    s = ''.join(['0', '2', '4', '6', '8'][d] for d in digits)
    print(s)