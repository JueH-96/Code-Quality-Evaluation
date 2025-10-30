def main():
    n = int(input().strip())
    points = []
    for _ in range(n):
        data = input().split()
        x = int(data[0])
        y = int(data[1])
        points.append((x, y))
    
    results = []
    for i in range(n):
        max_d_sq = -1
        candidate_index = None
        for j in range(n):
            if i == j:
                continue
            x1, y1 = points[i]
            x2, y2 = points[j]
            d_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if candidate_index is None:
                max_d_sq = d_sq
                candidate_index = j
            else:
                if d_sq > max_d_sq:
                    max_d_sq = d_sq
                    candidate_index = j
                elif d_sq == max_d_sq:
                    if j < candidate_index:
                        candidate_index = j
        results.append(str(candidate_index + 1))
    
    print("
".join(results))

if __name__ == "__main__":
    main()