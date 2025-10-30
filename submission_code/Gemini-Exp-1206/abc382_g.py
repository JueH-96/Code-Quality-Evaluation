def solve():
    k, sx, sy, tx, ty = map(int, input().split())

    def get_tile(x, y, k):
        i = x // k
        j = y // k
        if (i % 2 == j % 2):
            kk = (y % k)
        else:
            kk = (x % k)
        return (i, j, kk)

    start_tile = get_tile(sx, sy, k)
    end_tile = get_tile(tx, ty, k)

    if start_tile == end_tile:
        print(0)
        return

    q = [(start_tile, 0)]
    visited = {start_tile}

    while q:
        curr_tile, dist = q.pop(0)
        i, j, kk = curr_tile

        neighbors = []
        
        # Same parity
        if (i % 2 == j % 2):
            neighbors.append((i + 1, j, kk))
            neighbors.append((i - 1, j, kk))
            if kk > 0:
                neighbors.append((i, j, kk - 1))
            if kk < k - 1:
                neighbors.append((i, j, kk + 1))
        else:
            neighbors.append((i, j + 1, kk))
            neighbors.append((i, j - 1, kk))
            if kk > 0:
                neighbors.append((i, j, kk - 1))
            if kk < k - 1:
                neighbors.append((i, j, kk + 1))
        
        
        for neighbor in neighbors:
            ni, nj, nkk = neighbor
            
            if (ni % 2 == nj % 2):
                if (i % 2 == j % 2):
                    if ni == i + 1 and nj == j and nkk == kk:
                        pass
                    elif ni == i - 1 and nj == j and nkk == kk:
                        pass
                    elif ni == i and nj == j and nkk == kk + 1 and kk < k -1:
                        pass
                    elif ni == i and nj == j and nkk == kk - 1 and kk > 0:
                        pass
                    else:
                        continue
                else:
                    continue
            else:
                if (i % 2 != j % 2):
                    if ni == i and nj == j + 1 and nkk == kk:
                        pass
                    elif ni == i and nj == j - 1 and nkk == kk:
                        pass
                    elif ni == i and nj == j and nkk == kk + 1 and kk < k -1:
                        pass
                    elif ni == i and nj == j and nkk == kk - 1 and kk > 0:
                        pass
                    else:
                        continue
                else:
                    continue

            if neighbor == end_tile:
                print(dist + 1)
                return
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, dist + 1))

t = int(input())
for _ in range(t):
    solve()