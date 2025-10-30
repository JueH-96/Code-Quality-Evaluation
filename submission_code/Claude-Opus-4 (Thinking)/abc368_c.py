N = int(input())
healths = list(map(int, input().split()))

T = 0

for h in healths:
    while h > 0:
        T += 1
        if T % 3 == 0:
            h -= 3
        else:
            h -= 1

print(T)