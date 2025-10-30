# YOUR CODE HERE
import sys

def are_similar_chars(x, y):
    if x == y:
        return True
    if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    return False

def are_similar_strings(S, T):
    for x, y in zip(S, T):
        if not are_similar_chars(x, y):
            return False
    return True

input = sys.stdin.read().split()
N = int(input[0])
S = input[1]
T = input[2]

if are_similar_strings(S, T):
    print("Yes")
else:
    print("No")