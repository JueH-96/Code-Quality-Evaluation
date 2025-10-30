import sys

def main() -> None:
    # Read input
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    H = list(map(int, data[1:]))

    # rem_damage[s][left] :
    #   s    -> remainder of the first attack number modulo 3
    #   left -> number (0/1/2) of extra attacks after complete groups of three
    rem_damage = [
        [0, 3, 4],   # s == 0  (first attack is 3-damage)
        [0, 1, 2],   # s == 1  (first attack is 1-damage)
        [0, 1, 4]    # s == 2  (first attack is 1-damage, second 3-damage)
    ]

    T = 0                # number of attacks performed so far
    ceil5 = lambda x: (x + 4) // 5   # fast ceil(x / 5)

    for h in H:
        s = (T + 1) % 3              # remainder of the next attack index
        best = 10**20                # large initial value

        for left in (0, 1, 2):
            extra_damage = rem_damage[s][left]

            if h <= extra_damage:
                cycles = 0
            else:
                cycles = ceil5(h - extra_damage)   # full groups of 3 attacks

            attacks_needed = 3 * cycles + left
            if attacks_needed < best:
                best = attacks_needed

        T += best                     # update total attack count

    print(T)

if __name__ == "__main__":
    main()