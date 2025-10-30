import sys

def main():
    input = sys.stdin.readline
    N, Q = map(int, input().split())
    # Hx[i], Hy[i] = head position after i moves; start at (1,0)
    Hx = [1]
    Hy = [0]
    moves = 0
    out = []
    
    for _ in range(Q):
        parts = input().split()
        if parts[0] == '1':
            # Move the head
            C = parts[1]
            if C == 'R':
                dx, dy = 1, 0
            elif C == 'L':
                dx, dy = -1, 0
            elif C == 'U':
                dx, dy = 0, 1
            else:  # 'D'
                dx, dy = 0, -1
            Hx.append(Hx[moves] + dx)
            Hy.append(Hy[moves] + dy)
            moves += 1
        else:
            # Query segment p
            p = int(parts[1])
            delay = p - 1
            if moves >= delay:
                idx = moves - delay
                x, y = Hx[idx], Hy[idx]
            else:
                # Still on the initial horizontal line
                x, y = p - moves, 0
            out.append(f"{x} {y}")
    
    sys.stdout.write("
".join(out))

main()