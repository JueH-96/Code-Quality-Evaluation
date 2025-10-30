from functools import lru_cache

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    data = list(map(int, input[1:]))
    cards = []
    for i in range(n):
        a = data[2*i]
        b = data[2*i + 1]
        cards.append((a, b))
    
    @lru_cache(maxsize=None)
    def dfs(mask):
        remaining = []
        for i in range(n):
            if mask & (1 << i):
                remaining.append(i)
        m = len(remaining)
        for i in range(m):
            for j in range(i+1, m):
                a_i, b_i = cards[remaining[i]]
                a_j, b_j = cards[remaining[j]]
                if a_i == a_j or b_i == b_j:
                    new_mask = mask ^ ( (1 << remaining[i]) | (1 << remaining[j]) )
                    if not dfs(new_mask):
                        return True
        return False
    
    result = dfs((1 << n) - 1)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()