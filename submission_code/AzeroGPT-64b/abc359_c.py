import sys
input = sys.stdin.readline

def main():
    Sx, Sy = map(int, input().split())
    Tx, Ty = map(int, input().split())
    Sx = Sx * 2 + 1
    Sy = Sy * 2 + 1
    Tx = Tx * 2 + 1
    Ty = Ty * 2 + 1
    if Sx % 2 == 0:
        Sx -= 1
    if Sy % 2 == 0:
        Sy -= 1
    if Tx % 2 == 0:
        Tx -= 1
    if Ty % 2 == 0:
        Ty -= 1
    h, v = abs(Tx - Sx), abs(Ty - Sy)
    if h % 2 == 0: v += 2
    cost = v // 4 + h // 2
    print((cost))

def __starting_point():
    main()

__starting_point()