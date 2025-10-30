import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    from functools import lru_cache

    data = sys.stdin.read().split()
    N, M, L = map(int, data[:3])
    idx = 3
    A = list(map(int, data[idx:idx+N])); idx+=N
    B = list(map(int, data[idx:idx+M])); idx+=M
    C = list(map(int, data[idx:idx+L])); idx+=L

    A.sort()
    B.sort()
    C.sort()

    @lru_cache(None)
    def dfs(a_hand, b_hand, table, turn):
        # turn: 0 for Takahashi, 1 for Aoki
        if turn == 0:
            hand = list(a_hand)
        else:
            hand = list(b_hand)
        # If no cards to play, lose
        if not hand:
            return False
        table_list = list(table)
        # Try all moves
        for i, card in enumerate(hand):
            # remove card from hand, play it
            new_table = table_list + [card]
            # prepare hand after playing
            new_hand_base = hand[:i] + hand[i+1:]
            # find possible pickups
            # any table card less than card BEFORE we added card
            # that's just table_list
            smaller = [b for b in table_list if b < card]
            # Option 1: pick nothing
            nh = new_hand_base[:]  # copy
            nt = new_table[:]
            # assign to next state
            if turn == 0:
                na = tuple(sorted(nh))
                nb = b_hand
            else:
                na = a_hand
                nb = tuple(sorted(nh))
            nt_tup = tuple(sorted(nt))
            if not dfs(na, nb, nt_tup, 1-turn):
                return True
            # Option 2: pick one smaller, if any
            for pick in smaller:
                # remove one instance of pick from new_table and add to hand
                nt2 = nt[:]
                nt2.remove(pick)
                nh2 = new_hand_base[:] + [pick]
                if turn == 0:
                    na2 = tuple(sorted(nh2))
                    nb2 = b_hand
                else:
                    na2 = a_hand
                    nb2 = tuple(sorted(nh2))
                nt2_tup = tuple(sorted(nt2))
                if not dfs(na2, nb2, nt2_tup, 1-turn):
                    return True
        # if no winning move
        return False

    init_a = tuple(A)
    init_b = tuple(B)
    init_c = tuple(C)
    ok = dfs(init_a, init_b, init_c, 0)
    print("Takahashi" if ok else "Aoki")

if __name__ == "__main__":
    main()