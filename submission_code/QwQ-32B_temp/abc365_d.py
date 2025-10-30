import sys

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    prev_R = -float('inf')
    prev_P = -float('inf')
    prev_S = -float('inf')

    first_char = S[0]
    for m in ['R', 'P', 'S']:
        a = first_char
        if m == a:
            points = 0
            valid = True
        elif ( (m == 'R' and a == 'S') or
               (m == 'S' and a == 'P') or
               (m == 'P' and a == 'R') ):
            points = 1
            valid = True
        else:
            valid = False
        if valid:
            if m == 'R':
                prev_R = points
            elif m == 'P':
                prev_P = points
            else:
                prev_S = points

    for i in range(1, N):
        current_char = S[i]
        new_R = -float('inf')
        new_P = -float('inf')
        new_S = -float('inf')
        for m in ['R', 'P', 'S']:
            a = current_char
            valid = False
            points = 0
            if m == a:
                points = 0
                valid = True
            elif ( (m == 'R' and a == 'S') or
                   (m == 'S' and a == 'P') or
                   (m == 'P' and a == 'R') ):
                points = 1
                valid = True
            else:
                valid = False
            if valid:
                if m == 'R':
                    best_prev = max(prev_P, prev_S)
                elif m == 'P':
                    best_prev = max(prev_R, prev_S)
                else:  # S
                    best_prev = max(prev_R, prev_P)
                total = best_prev + points
                if m == 'R':
                    new_R = total
                elif m == 'P':
                    new_P = total
                else:
                    new_S = total
        prev_R, prev_P, prev_S = new_R, new_P, new_S

    max_val = max(prev_R, prev_P, prev_S)
    print(int(max_val))

if __name__ == "__main__":
    main()