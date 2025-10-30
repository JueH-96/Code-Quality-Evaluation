from typing import Tuple, List
from functools import reduce as rd
from operator import itemgetter
from itertools import starmap


def main() -> None:
    N: int = int(input())
    lst: List[Tuple[int, int]] = sorted(tuple(starmap(int, (input().split() for _ in range(N)))), key=itemgetter(1))
    print(rd(lambda acc: (acc[1] + max(acc[0], lst[acc[2]][0]), lst[acc[2]][1]) if acc[2] < N else (0, acc[1]), [(N, 0, i) for i in reversed(range(N))], (0, 0))[1])


if __name__ == '__main__':
    main()