# Read input
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    # Combine positions and counts, and sort by position
    stones = sorted(zip(X, A), key=lambda x: x[0])

    total_ops = 0
    prev_pos = 1
    if stones:
        carryover = stones[0][1] -1
    else:
        carryover = 0

    # If the first stone is not at position 1, handle the initial gap
    if stones:
        if stones[0][0] > 1:
            gap = stones[0][0] -1
            if carryover < gap:
                print(-1)
                return
            total_ops += gap * (gap +1) //2
            carryover -= gap
    else:
        # No stones, but N >=1, impossible
        print(-1)
        return

    for i in range(1, M):
        current_pos, current_A = stones[i]
        prev_pos, prev_A = stones[i-1]
        gap = current_pos - prev_pos -1
        if carryover < gap:
            print(-1)
            return
        total_ops += gap * (gap +1) //2
        carryover -= gap
        carryover += current_A -1

    # Handle the gap after the last stone to N
    last_pos, last_A = stones[-1]
    gap = N - last_pos
    if carryover < gap:
        print(-1)
        return
    total_ops += gap * (gap +1) //2
    carryover -= gap

    if carryover ==0:
        print(total_ops)
    else:
        print(-1)

if __name__ == "__main__":
    main()