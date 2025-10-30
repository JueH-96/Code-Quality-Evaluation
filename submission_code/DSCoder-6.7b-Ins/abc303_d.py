def solve():
    X, Y, Z = map(int, input().split())
    S = input()

    time = 0
    caps_on = False
    screen = ""

    for char in S:
        if char == 'A':
            if caps_on:
                time += X
                screen += 'A'
            else:
                time += Y
                screen += 'A'
        else:
            if caps_on:
                time += X
                screen += 'a'
            else:
                time += Y
                screen += 'A'
                caps_on = True
                time += Z

    print(time)

solve()