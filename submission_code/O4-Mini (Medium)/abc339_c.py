import sys
import threading

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = map(int, data[1:])

    cur = 0
    min_pref = 0  # track the minimum prefix sum seen
    for a in A:
        cur += a
        if cur < min_pref:
            min_pref = cur

    # The initial passengers X must be at least -min_pref to keep counts non-negative,
    # and at least 0. So X = max(0, -min_pref).
    X = -min_pref
    if X < 0:
        X = 0

    # Final count is total sum + X
    result = cur + X
    print(result)

if __name__ == "__main__":
    main()