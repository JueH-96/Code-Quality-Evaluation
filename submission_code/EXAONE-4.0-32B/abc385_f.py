import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    buildings = []
    for i in range(1, n + 1):
        x, h = map(int, data[i].split())
        buildings.append((x, h))
    
    hull = []
    T_list = []
    NEG_INF = -10**20
    
    for j in range(n):
        x_j, h_j = buildings[j]
        if j == 0:
            T_list.append(NEG_INF)
            hull.append((x_j, h_j))
        else:
            low = 0
            high = len(hull) - 1
            while high - low > 2:
                mid1 = low + (high - low) // 3
                mid2 = high - (high - low) // 3
                x1, h1 = hull[mid1]
                x2, h2 = hull[mid2]
                num1 = h1 * x_j - h_j * x1
                den1 = x_j - x1
                num2 = h2 * x_j - h_j * x2
                den2 = x_j - x2
                if num1 * den2 < num2 * den1:
                    low = mid1
                else:
                    high = mid2
            
            best_val = NEG_INF
            for idx in range(low, high + 1):
                x_i, h_i = hull[idx]
                num = h_i * x_j - h_j * x_i
                den = x_j - x_i
                val = num / den
                if val > best_val:
                    best_val = val
            T_list.append(best_val)
            
            while len(hull) >= 2:
                Ax, Ah = hull[-2]
                Bx, Bh = hull[-1]
                Cx, Ch = x_j, h_j
                cross = (Bx - Ax) * (Ch - Ah) - (Bh - Ah) * (Cx - Ax)
                if cross <= 0:
                    hull.pop()
                else:
                    break
            hull.append((x_j, h_j))
    
    non_neg_T = [t for t in T_list if t >= 0]
    if not non_neg_T:
        print(-1)
    else:
        ans = max(non_neg_T)
        print("{:.15f}".format(ans))

if __name__ == "__main__":
    main()