import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    if N == 0:
        m = min(H, W)
        S1 = m * (m + 1) // 2
        S2 = m * (m + 1) * (2 * m + 1) // 6
        total = m * (H + 1) * (W + 1) - (H + W + 2) * S1 + S2
        print(total)
    else:
        hole_set = set()
        idx = 3
        for i in range(N):
            a = int(data[idx])
            b = int(data[idx + 1])
            idx += 2
            hole_set.add((a - 1, b - 1))
        
        prev = [0] * W
        curr = [0] * W
        total = 0
        
        for i in range(H):
            for j in range(W):
                if (i, j) in hole_set:
                    curr[j] = 0
                else:
                    if i == 0 or j == 0:
                        curr[j] = 1
                    else:
                        curr[j] = min(prev[j], curr[j - 1], prev[j - 1]) + 1
                total += curr[j]
            prev, curr = curr, prev
        
        print(total)

if __name__ == "__main__":
    main()