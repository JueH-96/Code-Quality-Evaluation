import heapq

h, w = map(int, input().split())
grid = []
s = None
t = None
for i in range(h):
    row = input().strip()
    grid.append(row)
    for j in range(w):
        if row[j] == 'S':
            s = (i, j)
        if row[j] == 'T':
            t = (i, j)
n = int(input())
medicines = {}
for _ in range(n):
    r, c, e = map(int, input().split())
    r -= 1
    c -= 1
    medicines[(r, c)] = e

if s not in medicines:
    print("No")
    exit()

max_energy = [[-1 for _ in range(w)] for _ in range(h)]
heap = []
s_i, s_j = s
initial_energy = medicines[s]
max_energy[s_i][s_j] = initial_energy
heapq.heappush(heap, (-initial_energy, s_i, s_j))

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

while heap:
    current_neg, i, j = heapq.heappop(heap)
    current = -current_neg

    if (i, j) == t:
        print("Yes")
        exit()

    if current < max_energy[i][j]:
        continue

    for di, dj in directions:
        ni = i + di
        nj = j + dj
        if 0 <= ni < h and 0 <= nj < w:
            if grid[ni][nj] == '#':
                continue
            if current >= 1:
                new_energy = current - 1
                if new_energy > max_energy[ni][nj]:
                    max_energy[ni][nj] = new_energy
                    heapq.heappush(heap, (-new_energy, ni, nj))

    if (i, j) in medicines:
        e_i = medicines[(i, j)]
        if e_i > current:
            if e_i > max_energy[i][j]:
                max_energy[i][j] = e_i
                heapq.heappush(heap, (-e_i, i, j))

print("No")