# YOUR CODE HERE
N = int(input())
s = set()
max_a = int(N ** 0.5) + 1
for a in range(2, max_a):
    x = a * a
    while x <= N:
        s.add(x)
        if x > N // a:
            break
        x *= a
print(len(s) + 1)