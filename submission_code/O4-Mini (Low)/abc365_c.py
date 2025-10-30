import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    total_A = sum(A)
    # If sum of all A_i already within budget, x can be arbitrarily large
    if total_A <= M:
        print("infinite")
        return

    # Otherwise binary search for the maximum x
    low, high = 0, max(A)
    while low < high:
        mid = (low + high + 1) // 2
        s = 0
        # compute sum of min(mid, A_i)
        for a in A:
            if a < mid:
                s += a
            else:
                s += mid
            # early break if we already exceed M
            if s > M:
                break
        if s <= M:
            low = mid
        else:
            high = mid - 1

    print(low)

if __name__ == "__main__":
    main()