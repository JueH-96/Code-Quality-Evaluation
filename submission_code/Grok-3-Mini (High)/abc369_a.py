# YOUR CODE HERE
A, B = map(int, input().split())
s = set()
s.add(2 * A - B)
s.add(2 * B - A)
if (A + B) % 2 == 0:
    s.add((A + B) // 2)
print(len(s))