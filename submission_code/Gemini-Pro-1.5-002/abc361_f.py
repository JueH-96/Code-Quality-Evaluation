# YOUR CODE HERE
def solve():
    n = int(input())
    s = set()
    for b in range(2, 61):
        a = 1
        while True:
            x = a**b
            if x > n:
                break
            s.add(x)
            a += 1
    print(len(s))

solve()