# YOUR CODE HERE
def min_time_to_type(x, y, z, s):
    caps_on = False
    time = 0
    for char in s:
        if char.isupper():
            if caps_on:
                time += x
            else:
                time += min(y, z + x)
                caps_on = not caps_on
        else:
            if caps_on:
                time += min(y, z + x)
                caps_on = not caps_on
            else:
                time += x
    return time

x, y, z = map(int, input().split())
s = input().strip()
print(min_time_to_type(x, y, z, s))