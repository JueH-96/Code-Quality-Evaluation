def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))

    sumA = sum(A)
    minA = min(A)
    maxA = max(A)

    ans = -1
    # try every possible score x in [0,100]
    for x in range(101):
        total = sumA + x
        mn = min(minA, x)
        mx = max(maxA, x)
        final_grade = total - mn - mx
        if final_grade >= X:
            ans = x
            break

    print(ans)

main()