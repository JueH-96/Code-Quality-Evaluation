# YOUR CODE HERE
S1, S2 = input().split()
T1, T2 = input().split()

def distance(p1, p2):
    if (p1, p2) == ('A', 'C') or (p1, p2) == ('C', 'A') or (p1, p2) == ('E', 'C') or (p1, p2) == ('C', 'E'):
        return 1
    elif (p1, p2) == ('A', 'B') or (p1, p2) == ('B', 'A') or (p1, p2) == ('B', 'D') or (p1, p2) == ('D', 'B') or (p1, p2) == ('D', 'E') or (p1, p2) == ('E', 'D'):
        return 2
    else:
        return 3

if distance(S1, S2) == distance(T1, T2):
    print('Yes')
else:
    print('No')