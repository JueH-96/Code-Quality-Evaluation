import sys

def main():
    n, t = map(int, sys.stdin.readline().split())
    a_list = list(map(int, sys.stdin.readline().split()))
    
    rows = [0] * (n + 1)
    cols = [0] * (n + 1)
    main_diag = 0
    anti_diag = 0
    
    for turn in range(1, t + 1):
        a = a_list[turn - 1]
        val = a - 1
        i = (val // n) + 1
        j = (val % n) + 1
        
        # Update row and check
        rows[i] += 1
        if rows[i] == n:
            print(turn)
            return
        
        # Update column and check
        cols[j] += 1
        if cols[j] == n:
            print(turn)
            return
        
        # Check main diagonal
        if i == j:
            main_diag += 1
            if main_diag == n:
                print(turn)
                return
        
        # Check anti-diagonal
        if i + j == n + 1:
            anti_diag += 1
            if anti_diag == n:
                print(turn)
                return
    
    # If no bingo after all turns
    print(-1)

if __name__ == "__main__":
    main()