import sys
from collections import defaultdict

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))

    # frequency to the right of current position (initially whole array)
    right = defaultdict(int)
    for v in A:
        right[v] += 1

    left = defaultdict(int)           # frequency to the left of current position
    sum_pairs = 0                     # Σ left[v] * right[v]
    answer = 0

    for v in A:                       # j moves from 1 .. N
        # 1. j leaves the right side
        right[v] -= 1
        l = left[v]
        r = right[v]

        # update sum_pairs after right[v] decreased by 1
        sum_pairs -= l

        # 2. count triples whose middle index is current j
        #    valid ones exclude the value equal to A_j
        answer += sum_pairs - l * r

        # 3. j enters the left side
        left[v] += 1
        # update sum_pairs after left[v] increased by 1
        sum_pairs += r

    print(answer)

if __name__ == "__main__":
    main()