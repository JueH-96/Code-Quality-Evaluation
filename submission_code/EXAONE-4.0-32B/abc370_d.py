import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    H = int(data[0])
    W = int(data[1])
    Q = int(data[2])
    n = H * W
    active = [True] * n
    left_link = [-1] * n
    right_link = [-1] * n
    up_link = [-1] * n
    down_link = [-1] * n
    
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if j > 0:
                left_link[idx] = i * W + (j - 1)
            else:
                left_link[idx] = -1
                
            if j < W - 1:
                right_link[idx] = i * W + (j + 1)
            else:
                right_link[idx] = -1
                
            if i > 0:
                up_link[idx] = (i - 1) * W + j
            else:
                up_link[idx] = -1
                
            if i < H - 1:
                down_link[idx] = (i + 1) * W + j
            else:
                down_link[idx] = -1
                
    total_walls = H * W
    queries = []
    index = 3
    for i in range(Q):
        R = int(data[index]); C = int(data[index+1]); index += 2
        queries.append((R-1, C-1))
        
    def get_next_left(i, j):
        idx0 = i * W + j
        candidate = left_link[idx0]
        while candidate != -1 and not active[candidate]:
            next_candidate = left_link[candidate]
            left_link[idx0] = next_candidate
            candidate = next_candidate
        return candidate
        
    def get_next_right(i, j):
        idx0 = i * W + j
        candidate = right_link[idx0]
        while candidate != -1 and not active[candidate]:
            next_candidate = right_link[candidate]
            right_link[idx0] = next_candidate
            candidate = next_candidate
        return candidate
        
    def get_next_up(i, j):
        idx0 = i * W + j
        candidate = up_link[idx0]
        while candidate != -1 and not active[candidate]:
            next_candidate = up_link[candidate]
            up_link[idx0] = next_candidate
            candidate = next_candidate
        return candidate
        
    def get_next_down(i, j):
        idx0 = i * W + j
        candidate = down_link[idx0]
        while candidate != -1 and not active[candidate]:
            next_candidate = down_link[candidate]
            down_link[idx0] = next_candidate
            candidate = next_candidate
        return candidate
        
    for (r, c) in queries:
        idx0 = r * W + c
        if active[idx0]:
            active[idx0] = False
            total_walls -= 1
            left_nbr = left_link[idx0]
            right_nbr = right_link[idx0]
            up_nbr = up_link[idx0]
            down_nbr = down_link[idx0]
            
            if left_nbr != -1:
                right_link[left_nbr] = right_nbr
            if right_nbr != -1:
                left_link[right_nbr] = left_nbr
            if up_nbr != -1:
                down_link[up_nbr] = down_nbr
            if down_nbr != -1:
                up_link[down_nbr] = up_nbr
        else:
            candidates = set()
            cand_left = get_next_left(r, c)
            if cand_left != -1:
                candidates.add(cand_left)
            cand_right = get_next_right(r, c)
            if cand_right != -1:
                candidates.add(cand_right)
            cand_up = get_next_up(r, c)
            if cand_up != -1:
                candidates.add(cand_up)
            cand_down = get_next_down(r, c)
            if cand_down != -1:
                candidates.add(cand_down)
                
            for idx in candidates:
                if active[idx]:
                    active[idx] = False
                    total_walls -= 1
                    i1 = idx // W
                    j1 = idx % W
                    left_nbr = left_link[idx]
                    right_nbr = right_link[idx]
                    up_nbr = up_link[idx]
                    down_nbr = down_link[idx]
                    
                    if left_nbr != -1:
                        right_link[left_nbr] = right_nbr
                    if right_nbr != -1:
                        left_link[right_nbr] = left_nbr
                    if up_nbr != -1:
                        down_link[up_nbr] = down_nbr
                    if down_nbr != -1:
                        up_link[down_nbr] = up_nbr
                        
    print(total_walls)

if __name__ == "__main__":
    main()