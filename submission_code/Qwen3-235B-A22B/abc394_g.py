import sys
import heapq

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1

    F = []
    for _ in range(H):
        row = list(map(int, input[ptr:ptr+W]))
        ptr += W
        F.append(row)
    
    Q = int(input[ptr])
    ptr += 1

    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    for _ in range(Q):
        A = int(input[ptr])-1
        ptr +=1
        B = int(input[ptr])-1
        ptr +=1
        Y = int(input[ptr])
        ptr +=1
        C = int(input[ptr])-1
        ptr +=1
        D = int(input[ptr])-1
        ptr +=1
        Z = int(input[ptr])
        ptr +=1

        if A == C and B == D:
            print(abs(Y - Z))
            continue
        
        F_ab = F[A][B]
        F_cd = F[C][D]

        max_x = [[0]*W for _ in range(H)]
        heap = []
        heapq.heappush(heap, (-F_ab, A, B))
        max_x[A][B] = F_ab
        found = False

        while heap:
            current_neg, i, j = heapq.heappop(heap)
            current_x = -current_neg

            if i == C and j == D:
                found = True
                break

            if current_x < max_x[i][j]:
                continue

            for di, dj in directions:
                ni = i + di
                nj = j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    new_x = min(current_x, F[ni][nj])
                    if new_x > max_x[ni][nj]:
                        max_x[ni][nj] = new_x
                        heapq.heappush(heap, (-new_x, ni, nj))
        
        x_path = max_x[C][D]
        x_upper = min(x_path, F_ab, F_cd)

        min_yz = min(Y, Z)
        if x_upper >= min_yz:
            ans = abs(Y - Z)
        else:
            ans = (Y + Z) - 2 * x_upper
        
        print(ans)

if __name__ == "__main__":
    main()