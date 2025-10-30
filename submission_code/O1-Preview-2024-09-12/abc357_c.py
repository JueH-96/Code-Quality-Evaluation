# YOUR CODE HERE
N = int(input())
s = 3 ** N

def is_black(i,j):
    while i > 0 or j > 0:
        if i % 3 ==1 and j%3 ==1:
            return False
        i //=3
        j //=3
    return True

for i in range(s):
    line = []
    for j in range(s):
        if is_black(i,j):
            line.append('#')
        else:
            line.append('.')
    print(''.join(line))