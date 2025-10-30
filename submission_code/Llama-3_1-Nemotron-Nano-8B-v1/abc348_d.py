import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1
    
    grid = []
    start = None
    goal = None
    for i in range(H):
        line = input[idx].strip(); idx +=1
        grid.append(line)
        for j in range(W):
            if line[j] == 'S':
                start = (i, j)
            if line[j] == 'T':
                goal = (i, j)
    
    N = int(input[idx]); idx +=1
    med_dict = {}
    for _ in range(N):
        R = int(input[idx])-1; idx +=1
        C = int(input[idx])-1; idx +=1
        E = int(input[idx]); idx +=1
        med_dict[(R,C)] = E
    
    max_energy = [[-1 for _ in range(W)] for _ in range(H)]
    sr, sc = start
    max_energy[sr][sc] = 0
    heap = []
    heapq.heappush(heap, (-0, sr, sc))  # using max-heap via negative
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    found = False
    
    while heap:
        current_e_neg, i, j = heapq.heappop(heap)
        current_e = -current_e_neg
        
        if (i, j) == goal:
            print("Yes")
            found = True
            break
        
        if current_e < max_energy[i][j]:
            continue
        
        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                cell = grid[ni][nj]
                if cell == '#':
                    continue
                new_e = current_e -1
                if new_e >=0 and new_e > max_energy[ni][nj]:
                    max_energy[ni][nj] = new_e
                    heapq.heappush(heap, (-new_e, ni, nj))
        
        if (i,j) in med_dict:
            e_med = med_dict[(i,j)]
            if current_e < e_med:
                if e_med > max_energy[i][j]:
                    max_energy[i][j] = e_med
                    heapq.heappush(heap, (-e_med, i, j))
    
    if not found:
        print("No")

if __name__ == "__main__":
    main()