def solve():
    cards = list(map(int, input().split()))
    counts = {}
    for card in cards:
        counts[card] = counts.get(card, 0) + 1
    
    rank_counts = list(counts.values())
    rank_counts.sort(reverse=True)
    
    if rank_counts == [3, 1] or rank_counts == [2, 2]:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()