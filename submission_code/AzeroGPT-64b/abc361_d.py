from functools import cache

def solve(S, T):
    N = len(S)
    @cache
    def dp(index, rem):
        if index == N:
            return 0 if rem == '' else float('inf')
        if S[index] != T[index]:
            return float('inf')
        return min(
            abs(index - rem.find(T[index], 0, rem.find(T[index]) + 2)) + 1 + dp(index + 1, rem[rem.find(T[index])+1:]),
            dp(index + 1, rem)
        )
    res = dp(0, T)
    return -1 if res == float('inf') else res

def main():
    N = int(input())
    S, T = input(), input()
    print(solve(S, T))

if __name__ == "__main__":
    main()