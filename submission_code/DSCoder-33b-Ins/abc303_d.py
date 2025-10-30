# YOUR CODE HERE

def solve(x, y, z, s):
    time = 0
    caps_lock_on = False
    for i in range(len(s)):
        if s[i] == 'A':
            if not caps_lock_on:
                time += x
                caps_lock_on = True
            else:
                time += y
        else:
            if caps_lock_on:
                time += x
            else:
                time += y
                caps_lock_on = True
        time += z
    return time

x, y, z = map(int, input().split())
s = input()
print(solve(x, y, z, s))