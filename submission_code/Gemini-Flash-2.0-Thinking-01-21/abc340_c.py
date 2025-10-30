memo = {}

def solve(n):
    if n < 2:
        return 0
    if n in memo:
        return memo[n]
    l = n // 2
    r = (n + 1) // 2
    result = n + solve(l) + solve(r)
    memo[n] = result
    return result

if __name__ == '__main__':
    n = int(input())
    result = solve(n)
    print(result)