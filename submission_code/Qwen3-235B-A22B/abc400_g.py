import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, K = int(input[ptr]), int(input[ptr+1])
        ptr += 2
        cakes = []
        for _ in range(N):
            x = int(input[ptr])
            y = int(input[ptr+1])
            z = int(input[ptr+2])
            ptr += 3
            cakes.append((x, y, z))
        
        pairs = []
        # Generate pairs for X dimension
        sorted_x = sorted(range(N), key=lambda i: (-cakes[i][0], -cakes[i][1], -cakes[i][2]))
        for i in range(K):
            a = 2 * i
            b = 2 * i + 1
            if b >= N:
                break
            u = sorted_x[a]
            v = sorted_x[b]
            s = cakes[u][0] + cakes[v][0]
            pairs.append((s, u, v))
        
        # Generate pairs for Y dimension
        sorted_y = sorted(range(N), key=lambda i: (-cakes[i][1], -cakes[i][0], -cakes[i][2]))
        for i in range(K):
            a = 2 * i
            b = 2 * i + 1
            if b >= N:
                break
            u = sorted_y[a]
            v = sorted_y[b]
            s = cakes[u][1] + cakes[v][1]
            pairs.append((s, u, v))
        
        # Generate pairs for Z dimension
        sorted_z = sorted(range(N), key=lambda i: (-cakes[i][2], -cakes[i][0], -cakes[i][1]))
        for i in range(K):
            a = 2 * i
            b = 2 * i + 1
            if b >= N:
                break
            u = sorted_z[a]
            v = sorted_z[b]
            s = cakes[u][2] + cakes[v][2]
            pairs.append((s, u, v))
        
        # Sort pairs descending by sum
        pairs.sort(reverse=True, key=lambda x: x[0])
        
        used = [False] * N
        ans = 0
        cnt = 0
        for s, u, v in pairs:
            if not used[u] and not used[v]:
                ans += s
                used[u] = used[v] = True
                cnt += 1
                if cnt == K:
                    break
        print(ans)

if __name__ == "__main__":
    main()