def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    points = []
    index = 1
    for i in range(n):
        x = int(data[index])
        y = int(data[index+1])
        index += 2
        points.append((x, y))
    
    results = []
    for i in range(n):
        max_dist_sq = -1
        candidate_id = None
        for j in range(n):
            if i == j:
                continue
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + dy*dy
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                candidate_id = j + 1
            elif dist_sq == max_dist_sq:
                if j + 1 < candidate_id:
                    candidate_id = j + 1
        results.append(candidate_id)
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()