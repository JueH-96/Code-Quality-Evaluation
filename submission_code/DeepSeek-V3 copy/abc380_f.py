from functools import lru_cache

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    C = list(map(int, data[3+N+M:3+N+M+L]))
    
    # Convert lists to tuples for hashing in memoization
    A_tuple = tuple(sorted(A))
    B_tuple = tuple(sorted(B))
    C_tuple = tuple(sorted(C))
    
    @lru_cache(maxsize=None)
    def game(takahashi, aoki, table):
        # Takahashi's turn
        if not takahashi:
            return False  # Takahashi loses
        # Try all possible moves for Takahashi
        for i in range(len(takahashi)):
            new_takahashi = list(takahashi)
            card = new_takahashi.pop(i)
            new_table = list(table)
            new_table.append(card)
            new_table.sort()
            # Check if Takahashi can take a card
            taken = False
            for j in range(len(new_table)):
                if new_table[j] < card:
                    taken = True
                    new_takahashi.append(new_table[j])
                    new_table.pop(j)
                    break
            # Recurse with Aoki's turn
            if not game(tuple(sorted(new_takahashi)), aoki, tuple(sorted(new_table))):
                return True
        # If no move leads to a win, Takahashi loses
        return False
    
    if game(A_tuple, B_tuple, C_tuple):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()