def solve():
    n = int(input())
    cards = []
    for _ in range(n):
        cards.append(list(map(int, input().split())))

    def can_remove(mask, i, j):
        if (mask >> i) & 1 == 0 or (mask >> j) & 1 == 0:
            return False
        return cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]

    def winner(mask):
        if mask == 0:
            return False
        
        for i in range(n):
            for j in range(i + 1, n):
                if can_remove(mask, i, j):
                    new_mask = mask ^ (1 << i) ^ (1 << j)
                    if not winner(new_mask):
                        return True
        return False

    if winner((1 << n) - 1):
        print("Takahashi")
    else:
        print("Aoki")

solve()