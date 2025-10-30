def main():
    import sys
    data = sys.stdin.readline().split()
    A = int(data[0])
    M = int(data[1])
    L = int(data[2])
    R = int(data[3])

    # We want k such that A + k*M is in [L, R].
    # Equivalently k*M is in [L-A, R-A].
    l = L - A
    r = R - A

    # Smallest k: ceil(l / M). Since M > 0, we can do integer ceil as (l + M - 1)//M.
    k_min = (l + M - 1) // M
    # Largest k: floor(r / M) is just r // M in Python (floor division).
    k_max = r // M

    # Count of integers in [k_min, k_max]
    ans = k_max - k_min + 1
    if ans < 0:
        ans = 0

    print(ans)

if __name__ == "__main__":
    main()