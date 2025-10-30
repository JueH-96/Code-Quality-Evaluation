import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    presses = []
    for _ in range(N):
        a, s = input().split()
        presses.append((int(a), s))

    INF = 10**18
    # dp[l][r] = minimum fatigue when left hand at l, right hand at r
    # initialize: both hands can start anywhere with 0 fatigue
    dp = [[0] * 101 for _ in range(101)]

    for a, hand in presses:
        new_dp = [[INF] * 101 for _ in range(101)]
        if hand == 'L':
            for l in range(1, 101):
                move_cost = abs(l - a)
                for r in range(1, 101):
                    cost = dp[l][r] + move_cost
                    if cost < new_dp[a][r]:
                        new_dp[a][r] = cost
        else:  # hand == 'R'
            for l in range(1, 101):
                for r in range(1, 101):
                    move_cost = abs(r - a)
                    cost = dp[l][r] + move_cost
                    if cost < new_dp[l][a]:
                        new_dp[l][a] = cost
        dp = new_dp

    # find the minimal fatigue over all end positions
    ans = INF
    for l in range(1, 101):
        for r in range(1, 101):
            if dp[l][r] < ans:
                ans = dp[l][r]

    print(ans)

if __name__ == "__main__":
    main()