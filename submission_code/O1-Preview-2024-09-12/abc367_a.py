# YOUR CODE HERE
A, B, C = map(int, input().split())

def is_asleep(t, B, C):
    if B < C:
        return B <= t < C
    else:
        return t >= B or t < C

if is_asleep(A, B, C):
    print('No')
else:
    print('Yes')