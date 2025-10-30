def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr +=1
    W = int(input[ptr])
    ptr +=1
    N = int(input[ptr])
    ptr +=1
    T = input[ptr]
    ptr +=1
    
    row_mask = []
    for _ in range(H):
        s = input[ptr]
        ptr +=1
        mask = 0
        for j in range(W):
            if s[j] == '.':
                mask |= 1 << j
        row_mask.append(mask)
    
    prev = list(row_mask)
    
    for move in T:
        curr = [0]*H
        if move == 'L':
            for i in range(H):
                curr[i] = (prev[i] >> 1) & row_mask[i]
        elif move == 'R':
            mask_w = (1 << W) -1
            for i in range(H):
                shifted = prev[i] << 1
                masked = shifted & mask_w
                curr[i] = masked & row_mask[i]
        elif move == 'U':
            for i in range(H):
                if i +1 < H:
                    curr[i] = prev[i+1] & row_mask[i]
                else:
                    curr[i] = 0
        elif move == 'D':
            for i in range(H):
                if i -1 >=0:
                    curr[i] = prev[i-1] & row_mask[i]
                else:
                    curr[i] = 0
        prev = curr
    
    ans = sum(bin(x).count('1') for x in prev)
    print(ans)

if __name__ == '__main__':
    main()