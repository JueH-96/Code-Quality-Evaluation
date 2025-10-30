# YOUR CODE HERE
def remove_abc(s):
    while 'ABC' in s:
        s = s.replace('ABC', '', 1)
    return s

s = input().strip()
print(remove_abc(s))