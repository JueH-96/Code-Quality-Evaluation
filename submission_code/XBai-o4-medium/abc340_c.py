import sys

def main():
    N = int(sys.stdin.readline())
    memo = {}
    
    def f(x):
        if x < 2:
            return 0
        if x in memo:
            return memo[x]
        a = x // 2
        b = x - a
        res = x + f(a) + f(b)
        memo[x] = res
        return res
    
    print(f(N))

if __name__ == "__main__":
    main()