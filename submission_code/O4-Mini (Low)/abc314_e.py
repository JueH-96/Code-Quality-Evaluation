import sys
def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    wheels = []
    for _ in range(N):
        data = list(map(int, input().split()))
        C = data[0]
        P = data[1]
        S = data[2:]
        wheels.append((C, P, S))

    # E[x] = expected cost to go from score x to >= M
    # E[M] = 0, for x<M:
    #   for each wheel i:
    #     let p0 = prob of S=0,  R = sum_{S>0} E[x+S]/P
    #     then E_i = (C + R)/(1-p0)
    #   E[x] = min_i E_i
    E = [0.0] * (M+1)
    E[M] = 0.0

    # fill from M-1 down to 0
    for x in range(M-1, -1, -1):
        best = float('inf')
        for C, P, S in wheels:
            cnt0 = 0
            R = 0.0
            invP = 1.0 / P
            for s in S:
                if s == 0:
                    cnt0 += 1
                else:
                    nx = x + s
                    if nx > M:
                        nx = M
                    R += E[nx] * invP
            p0 = cnt0 * invP
            # Avoid division by zero; problem guarantees sum S > 0 so not all zeros
            val = (C + R) / (1.0 - p0)
            if val < best:
                best = val
        E[x] = best

    # print E[0]
    print("{:.10f}".format(E[0]))


if __name__ == "__main__":
    main()