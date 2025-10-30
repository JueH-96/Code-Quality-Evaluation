def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    balls = []
    for x in a:
        balls.append(x)
        while len(balls) >= 2:
            if balls[-1] == balls[-2]:
                balls.pop()
                balls[-1] += 1
            else:
                break
    print(len(balls))

solve()