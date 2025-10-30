# YOUR CODE HERE
def solve():
    import sys
    import math

    input = sys.stdin.read().strip()
    L, R = map(int, input.split())
    x = L
    segments = []
    while x < R:
        if x == 0:
            trail = 60
        else:
            trail = (x & -x).bit_length() -1
        Rx = R - x
        if Rx ==0:
            log2_Rx = 0
        else:
            log2_Rx = Rx.bit_length() -1
        k = min(trail, log2_Rx)
        len_seg = 1 <<k
        segments.append( (x, x + len_seg) )
        x += len_seg
    print(len(segments))
    for l, r in segments:
        print(l, r)