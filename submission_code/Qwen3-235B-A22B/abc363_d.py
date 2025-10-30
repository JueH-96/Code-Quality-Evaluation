import sys

def main():
    N = int(sys.stdin.readline())
    if N <= 10:
        print(N - 1)
        return
    M = N - 10
    l = 2
    while True:
        d = (l - 1) // 2
        count = 9 * (10 ** d)
        if M > count:
            M -= count
            l += 1
        else:
            break
    # Determine the half part
    k = (l + 1) // 2
    half_start = 10 ** (k - 1)
    half_number = half_start + (M - 1)
    s = str(half_number)
    if l % 2 == 0:
        pal = s + s[::-1]
    else:
        pal = s + s[-2::-1]
    print(pal)

if __name__ == "__main__":
    main()