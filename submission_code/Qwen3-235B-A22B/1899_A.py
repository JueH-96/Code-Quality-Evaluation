from functools import lru_cache

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    t = int(data[0])
    cases = list(map(int, data[1:t+1]))

    @lru_cache(maxsize=None)
    def can_win(current, turn):
        if turn >= 10:
            return False
        is_vanya = (turn % 2) == 0
        if is_vanya:
            # Vanya's turn
            # Check for immediate win
            for delta in (-1, 1):
                if (current + delta) % 3 == 0:
                    return True
            # Check if any move leads to a forced win regardless of Vova's response
            for delta in (-1, 1):
                new_v = current + delta
                all_v_blocked = True
                for vdelta in (-1, 1):
                    next_n = new_v + vdelta
                    if not can_win(next_n, turn + 1):
                        all_v_blocked = False
                        break
                if all_v_blocked:
                    return True
            return False
        else:
            # Vova's turn
            # Check if there's a move that blocks all of Vanya's paths
            for delta in (-1, 1):
                new_v = current + delta
                all_vanya_blocked = True
                for vdelta in (-1, 1):
                    next_n = new_v + vdelta
                    if can_win(next_n, turn + 1):
                        all_vanya_blocked = False
                        break
                if all_vanya_blocked:
                    return True
            return False

    for n in cases:
        if can_win(n, 0):
            print("First")
        else:
            print("Second")

if __name__ == "__main__":
    main()