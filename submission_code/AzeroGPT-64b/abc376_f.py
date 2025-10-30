import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())

    # 1-indexed internal representation (because it's easier to not think)
    evens = [set() for _ in range(3)]
    odds = [set() for _ in range(3)]

    for i in range(Q):
        h, t = sys.stdin.readline().split()
        hand = 0 if h == "L" else 1
        (evens if int(t)%2 == 0 else odds)[hand].add(i)

    # init a clever queue to properly compute earliest availability (L, R, in order)
    even_LRQ = deque(zip([1, 2, 1], [2, 1, 1]))
    odd_LRQ = deque(zip([1, 2, 1], [2, 1, 1]))
    for _ in range(Q+1):
        if even_LRQ[0][1] > Q:
            even_LRQ.appendleft(even_LRQ.pop())
        if odd_LRQ[0][1] > Q:
            odd_LRQ.appendleft(odd_LRQ.pop())

    # figure out left & right hands' initial possesions (for solving).
    even_inventory = []
    odd_inventory = []
    even_LRQ[0] = (even_LRQ[0][0], -1)
    for i, j in zip([*evens[0], -1], [*evens[0][1:], N]):
        if i > -1:
            # i is the last of the smash lefts for even
            # j is the first of the hand over hands for even
            # N is the "ignore" because both 1 & N can be considered "even" for \
            # succession purposes.

            distance_right_even = j - i - 1 if j != N else N + 1 - i
            distance_left_even = (N + i + 1 - j) if j != N else i

            if distance_right_even < distance_left_even:
                inv = distance_right_even
            else:
                inv = distance_left_even + 1

            even_inventory.append([i, inv])

        even_LRQ.rotate(-1)

    odd_inventory = []
    odd_LRQ[0] = (odd_LRQ[0][0], -1)
    for i, j in zip([*odds[0], -1], [*odds[0][1:], N]):

        if i > -1:
            distance_right_odd = j - i - 1 if j != N else N + 1 - i
            distance_left_odd = (N + i + 1- j) if j != N else i

            if distance_right_odd < distance_left_odd:
                inv = distance_right_odd
            else:
                inv = distance_left_odd + 1

            odd_inventory.append([i, inv])

        odd_LRQ.rotate(-1)

    def solve(L, R, inventory, LRQ):
        operations = 0
        for idx in range(Q):
            if idx in L:
                while LRQ[1 - LRQ[0][0]][1] <= idx:
                    op = LRQ[0][0]  # L - 0, R - 1
                    thing = LRQ.popleft()[1]
                    LRQ.append((1 - LRQ[0][0], thing))

                    # add to operation count the "WORK TO DO": aka the inventory.
                    operations += inventory[thing - 1][1]
            elif idx in R:
                while LRQ[LRQ[0][0]][1] < idx:
                    op = LRQ[0][0]  # L - 0, R - 1
                    thing = LRQ.popleft()[1]
                    LRQ.append((1 - LRQ[0][0], thing))

                    # add to operation count the "WORK TO DO": aka the inventory.
                    operations += inventory[thing - 1][1]

        return operations

    evens += [[]]
    odds += [[]]
    # right hands, then left hands < interesting as hell
    ans = min(solve(evens[0], evens[1], even_inventory, even_LRQ), solve(odds[0], odds[1], odd_inventory, odd_LRQ))

    print(ans)
    return

if __name__ == "__main__":
    main()