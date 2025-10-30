def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    coords = []
    
    idx = 1
    for _ in range(N):
        x = int(data[idx]); y = int(data[idx+1])
        coords.append((x, y))
        idx += 2

    for i in range(N):
        best_idx = -1
        best_dist = -1
        x1, y1 = coords[i]
        for j in range(N):
            if i == j:
                continue
            x2, y2 = coords[j]
            dx = x1 - x2
            dy = y1 - y2
            dist = dx*dx + dy*dy  # compare squared distances to avoid floating-point error
            if dist > best_dist:
                best_dist = dist
                best_idx = j
            elif dist == best_dist and j < best_idx:
                best_idx = j
        print(best_idx + 1)

if __name__ == "__main__":
    main()