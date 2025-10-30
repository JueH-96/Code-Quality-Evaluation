# YOUR CODE HERE
N = int(input())

for x in range(N, 1000):
    h = x // 100
    t = (x // 10) % 10
    o = x % 10

    if h * t == o:
        print(x)
        break