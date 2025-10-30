import itertools

def solve(n: int, m: int, a: list):
    ans = 0
    mods = [0] * m
    
    for i in range(n):
        x = sum(a[:i]) % m
        ans += mods[m - x % m if x > n // 2 else x % m]
        mods[x % m] += 1

    return ans + m * (n - 1)
            
        
def main() -> None:
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    print(solve(n, m, a))

if __name__ == "__main__":
    main()