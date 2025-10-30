import sys

def main():
    sys.setrecursionlimit(1000000)
    N, A, X, Y = map(int, sys.stdin.read().split())
    memo = {0: 0.0}
    
    def e(n):
        if n in memo:
            return memo[n]
        if n == 0:
            memo[n] = 0.0
        else:
            option1 = X + e(n // A)
            sum_b = 0.0
            for b in range(1, 7):
                sum_b += e(n // b)
            option2 = (6 * Y + sum_b) / 6
            memo[n] = min(option1, option2)
        return memo[n]
    
    result = e(N)
    print("{0:.15f}".format(result))

if __name__ == "__main__":
    main()