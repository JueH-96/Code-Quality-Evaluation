import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # count occurrences
    cnt = [0] * (N + 1)
    mid_pos = [0] * (N + 1)

    # record the index of the 2nd occurrence for each number
    for idx, a in enumerate(A, start=1):
        cnt[a] += 1
        if cnt[a] == 2:
            mid_pos[a] = idx

    # pair (middle index, value) and sort
    pairs = [(mid_pos[i], i) for i in range(1, N + 1)]
    pairs.sort()

    # output the values in sorted order
    result = [str(i) for _, i in pairs]
    print(" ".join(result))

if __name__ == "__main__":
    main()