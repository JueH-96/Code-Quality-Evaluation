def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    processes = []
    for _ in range(N):
        A = int(input[idx])
        P = int(input[idx+1])
        B = int(input[idx+2])
        Q = int(input[idx+3])
        processes.append((A, P, B, Q))
        idx += 4

    low = 0
    high = 10**18
    answer = 0

    while low <= high:
        mid = (low + high) // 2
        total_cost = 0
        possible = True
        for (a, p, b, q) in processes:
            if mid == 0:
                cost = 0
            else:
                # Option 1: pure S
                s1 = (mid + a - 1) // a
                option1 = s1 * p

                # Option 2: pure T
                s2 = (mid + b - 1) // b
                option2 = s2 * q

                # Option 3: s = mid // a, then t
                s3 = mid // a
                rem3 = mid - a * s3
                if rem3 == 0:
                    option3 = s3 * p
                else:
                    t3 = (rem3 + b - 1) // b
                    option3 = s3 * p + t3 * q

                # Option4: t = mid // b, then s
                t4 = mid // b
                rem4 = mid - b * t4
                if rem4 == 0:
                    option4 = t4 * q
                else:
                    s4 = (rem4 + a - 1) // a
                    option4 = t4 * q + s4 * p

                cost = min(option1, option2, option3, option4)

            total_cost += cost
            if total_cost > X:
                possible = False
                break
        if possible:
            answer = mid
            low = mid + 1
        else:
            high = mid - 1

    print(answer)

if __name__ == '__main__':
    main()