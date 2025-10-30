import sys
from functools import lru_cache

def main():
    n = int(sys.stdin.readline())
    cards = []
    for _ in range(n):
        a, b = map(int, sys.stdin.readline().split())
        cards.append((a, b))
    
    @lru_cache(maxsize=None)
    def dfs(mask):
        active = [i for i in range(n) if (mask & (1 << i))]
        m = len(active)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = cards[active[i]]
                c, d = cards[active[j]]
                if a == c or b == d:
                    new_mask = mask ^ (1 << active[i]) ^ (1 << active[j])
                    if not dfs(new_mask):
                        return True
        return False
    
    result = dfs((1 << n) - 1)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()