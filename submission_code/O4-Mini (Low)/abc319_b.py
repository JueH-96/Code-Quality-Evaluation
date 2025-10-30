def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    # Collect divisors j of N in [1..9]
    js = [j for j in range(1, 10) if N % j == 0]
    js.sort()
    res = []
    for i in range(N + 1):
        ch = '-'
        for j in js:
            step = N // j
            if i % step == 0:
                ch = str(j)
                break
        res.append(ch)
    # Join and print the result
    print("".join(res))

if __name__ == "__main__":
    main()