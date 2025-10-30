import sys
from functools import lru_cache

def main():
    sys.setrecursionlimit(1 << 25)
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))
    
    A_sorted = tuple(sorted(A))
    B_sorted = tuple(sorted(B))
    C_sorted = tuple(sorted(C))
    
    @lru_cache(maxsize=None)
    def can_win(current_hand, other_hand, table):
        if not current_hand:
            return False
        for i in range(len(current_hand)):
            c = current_hand[i]
            new_current = current_hand[:i] + current_hand[i+1:]
            new_table = list(table)
            new_table.append(c)
            new_table_sorted = tuple(sorted(new_table))
            S = [x for x in new_table_sorted if x < c]
            if not S:
                if not can_win(other_hand, new_current, new_table_sorted):
                    return True
            else:
                for s in S:
                    s_index = new_table_sorted.index(s)
                    new_table_removed = list(new_table_sorted[:s_index]) + list(new_table_sorted[s_index+1:])
                    new_table_removed_sorted = tuple(sorted(new_table_removed))
                    new_current_with_s = list(new_current) + [s]
                    new_current_with_s_sorted = tuple(sorted(new_current_with_s))
                    if not can_win(other_hand, new_current_with_s_sorted, new_table_removed_sorted):
                        return True
                if not can_win(other_hand, new_current, new_table_sorted):
                    return True
        return False
    
    result = can_win(A_sorted, B_sorted, C_sorted)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()