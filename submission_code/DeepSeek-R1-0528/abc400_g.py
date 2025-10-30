import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        if index >= len(data):
            break
        N = int(data[index]); K = int(data[index+1]); index += 2
        cakes = []
        for i in range(N):
            x = int(data[index]); y = int(data[index+1]); z = int(data[index+2]); index += 3
            cakes.append([x, y, z])
        
        if K == 0:
            results.append("0")
            continue
            
        idx_x = sorted(range(N), key=lambda i: cakes[i][0], reverse=True)
        idx_y = sorted(range(N), key=lambda i: cakes[i][1], reverse=True)
        idx_z = sorted(range(N), key=lambda i: cakes[i][2], reverse=True)
        
        candidate_set = set()
        
        for idx_list in [idx_x, idx_y, idx_z]:
            for i in range(N):
                count = 0
                for j in idx_list:
                    if j == i:
                        continue
                    a, b = min(i, j), max(i, j)
                    x_sum = cakes[i][0] + cakes[j][0]
                    y_sum = cakes[i][1] + cakes[j][1]
                    z_sum = cakes[i][2] + cakes[j][2]
                    weight = max(x_sum, y_sum, z_sum)
                    candidate_set.add((a, b, weight))
                    count += 1
                    if count >= 4:
                        break
        
        candidate_list = list(candidate_set)
        candidate_list.sort(key=lambda edge: (-edge[2], -edge[0], -edge[1]))
        
        available = [True] * N
        total = 0
        count_edges = 0
        for edge in candidate_list:
            if count_edges >= K:
                break
            a, b, w = edge
            if available[a] and available[b]:
                total += w
                available[a] = False
                available[b] = False
                count_edges += 1
                
        results.append(str(total))
    
    print("
".join(results))

if __name__ == "__main__":
    main()