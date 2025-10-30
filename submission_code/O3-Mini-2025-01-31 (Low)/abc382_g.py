#!/usr/bin/env python3
def main():
    import sys,math
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    answers = []
    pos = 1
    for _ in range(t):
        K = int(data[pos]); pos += 1
        Sx = int(data[pos]); pos += 1
        Sy = int(data[pos]); pos += 1
        Tx = int(data[pos]); pos += 1
        Ty = int(data[pos]); pos += 1
        
        def get_tile(x, y):
            # point = (x+0.5, y+0.5)
            xP = x + 0.5
            yP = y + 0.5
            I = math.floor(xP / K)
            J = math.floor(yP / K)
            if (I + J) & 1 == 0:
                k_val = math.floor(yP - J * K)
            else:
                k_val = math.floor(xP - I * K)
            return I, J, k_val
        
        I_s, J_s, k_s = get_tile(Sx, Sy)
        I_t, J_t, k_t = get_tile(Tx, Ty)
        d_cells = abs(I_s - I_t) + abs(J_s - J_t)
        d_k = abs(k_s - k_t)
        moves = d_cells + max(0, d_k - d_cells + 1)
        answers.append(str(moves))
    sys.stdout.write("
".join(answers))

if __name__ == '__main__':
    main()