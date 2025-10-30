W, B = map(int, input().split())
S = 'wbwbwwbwbwbw' * 100
if 'w' * W + 'b' * B in S or 'b' * B + 'w' * W in S:
    print('Yes')
else:
    print('No')