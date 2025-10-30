def solve():
    n = int(input())
    cards = []
    for _ in range(n):
        a, b = map(int, input().split())
        cards.append((a, b))

    memo = {}

    def can_win(mask):
        if mask in memo:
            return memo[mask]

        possible_moves_exist = False
        remaining_cards = [i for i in range(n) if (mask >> i) & 1]
        num_remaining = len(remaining_cards)

        if num_remaining < 2:
            memo[mask] = False
            return False

        for i_idx in range(num_remaining):
            for j_idx in range(i_idx + 1, num_remaining):
                i = remaining_cards[i_idx]
                j = remaining_cards[j_idx]

                if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                    possible_moves_exist = True
                    new_mask = mask ^ (1 << i) ^ (1 << j)
                    if not can_win(new_mask):
                        memo[mask] = True
                        return True

        memo[mask] = False
        return False

    initial_mask = (1 << n) - 1
    if can_win(initial_mask):
        print("Takahashi")
    else:
        print("Aoki")

solve()