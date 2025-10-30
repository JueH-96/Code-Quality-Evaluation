from collections import defaultdict

def solve():
    N = int(input())
    cards = []
    for _ in range(N):
        a, b = map(int, input().split())
        cards.append((a, b))

    # Create a dictionary to store the indices of cards with the same front or back value
    front_dict = defaultdict(list)
    back_dict = defaultdict(list)
    for i, (a, b) in enumerate(cards):
        front_dict[a].append(i)
        back_dict[b].append(i)

    # Implement the game logic
    turn = 0
    while True:
        # Check if Takahashi can make a move
        found = False
        for i in front_dict:
            if len(front_dict[i]) >= 2:
                found = True
                break
        if not found:
            for i in back_dict:
                if len(back_dict[i]) >= 2:
                    found = True
                    break

        if not found:
            return "Aoki" if turn % 2 == 0 else "Takahashi"

        # Takahashi's turn
        if turn % 2 == 0:
            for i in front_dict:
                if len(front_dict[i]) >= 2:
                    idx1, idx2 = front_dict[i][:2]
                    front_dict[i].pop(0)
                    front_dict[i].pop(0)
                    back_dict[cards[idx1][1]].remove(idx1)
                    back_dict[cards[idx2][1]].remove(idx2)
                    break
            else:
                for i in back_dict:
                    if len(back_dict[i]) >= 2:
                        idx1, idx2 = back_dict[i][:2]
                        back_dict[i].pop(0)
                        back_dict[i].pop(0)
                        front_dict[cards[idx1][0]].remove(idx1)
                        front_dict[cards[idx2][0]].remove(idx2)
                        break
        # Aoki's turn
        else:
            for i in front_dict:
                if len(front_dict[i]) >= 2:
                    idx1, idx2 = front_dict[i][:2]
                    front_dict[i].pop(0)
                    front_dict[i].pop(0)
                    back_dict[cards[idx1][1]].remove(idx1)
                    back_dict[cards[idx2][1]].remove(idx2)
                    break
            else:
                for i in back_dict:
                    if len(back_dict[i]) >= 2:
                        idx1, idx2 = back_dict[i][:2]
                        back_dict[i].pop(0)
                        back_dict[i].pop(0)
                        front_dict[cards[idx1][0]].remove(idx1)
                        front_dict[cards[idx2][0]].remove(idx2)
                        break
        turn += 1