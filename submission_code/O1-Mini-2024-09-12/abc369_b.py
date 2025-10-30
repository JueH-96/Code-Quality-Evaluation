import sys

def main():
    import sys
    import math
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = []
    S = []
    for i in range(N):
        A.append(int(data[1 + 2*i]))
        S.append(data[2 + 2*i])

    INF = 10**18
    # Initialize previous DP with all possible initial positions
    prev = [[INF] * 101 for _ in range(101)]
    for l in range(1, 101):
        for r in range(1, 101):
            prev[l][r] = 0

    for i in range(N):
        curr = [[INF] * 101 for _ in range(101)]
        for l in range(1, 101):
            for r in range(1, 101):
                if prev[l][r] < INF:
                    if S[i] == 'L':
                        cost = abs(A[i] - l)
                        new_l = A[i]
                        new_r = r
                        if cost + prev[l][r] < curr[new_l][new_r]:
                            curr[new_l][new_r] = cost + prev[l][r]
                    else:
                        cost = abs(A[i] - r)
                        new_l = l
                        new_r = A[i]
                        if cost + prev[l][r] < curr[new_l][new_r]:
                            curr[new_l][new_r] = cost + prev[l][r]
        prev = curr

    result = INF
    for l in range(1, 101):
        for r in range(1, 101):
            if prev[l][r] < result:
                result = prev[l][r]
    print(result)

if __name__ == "__main__":
    main()