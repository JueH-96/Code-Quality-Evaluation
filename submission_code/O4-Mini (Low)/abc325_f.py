import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    D = list(map(int, input().split()))
    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())

    # Compute baseline x2 when using 0 type-1 sensors
    import math
    base_x2 = 0
    per_sec_max1 = []  # maximum number of type-1 sensors that can reduce the section
    gains = []         # marginal gains in x2 by spending one more type-1

    for d in D:
        # baseline x2
        x2_0 = (d + L2 - 1) // L2
        base_x2 += x2_0

        # max useful type-1 on this section
        max1 = (d + L1 - 1) // L1
        if max1 > K1:
            max1 = K1
        per_sec_max1.append(max1)

        prev_x2 = x2_0
        # For k = 1..max1, compute x2_k and gain = prev_x2 - x2_k
        for k in range(1, max1+1):
            rem = d - k*L1
            if rem <= 0:
                x2_k = 0
            else:
                x2_k = (rem + L2 - 1) // L2
            gain = prev_x2 - x2_k
            # gain >= 0
            gains.append(gain)
            prev_x2 = x2_k

    # Sort gains descending, we'll pick top U gains for U type-1 sensors
    gains.sort(reverse=True)
    # Build prefix sums up to K1
    P = [0] * (K1 + 1)
    running = 0
    for i in range(1, K1+1):
        if i-1 < len(gains):
            running += gains[i-1]
        # else running += 0
        P[i] = running

    INF = 10**40
    answer = INF

    # Try using U type-1 sensors (0..K1)
    for U in range(K1+1):
        # total reduction in x2
        red = P[U]
        req_x2 = base_x2 - red
        if req_x2 < 0:
            req_x2 = 0
        if req_x2 <= K2:
            cost = U * C1 + req_x2 * C2
            if cost < answer:
                answer = cost

    if answer >= INF:
        print(-1)
    else:
        print(answer)

if __name__ == "__main__":
    main()