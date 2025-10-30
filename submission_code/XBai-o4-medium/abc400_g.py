import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        K = int(input[ptr+1])
        ptr += 2
        X = []
        Y = []
        Z = []
        for i in range(N):
            x = int(input[ptr])
            y = int(input[ptr+1])
            z = int(input[ptr+2])
            X.append(x)
            Y.append(y)
            Z.append(z)
            ptr += 3
        
        # Generate the three sorted lists
        sorted_x = sorted( [(X[i], Y[i], Z[i], i) for i in range(N)], key=lambda t: (-t[0], -t[1], -t[2]) )
        sorted_y = sorted( [(X[i], Y[i], Z[i], i) for i in range(N)], key=lambda t: (-t[1], -t[0], -t[2]) )
        sorted_z = sorted( [(X[i], Y[i], Z[i], i) for i in range(N)], key=lambda t: (-t[2], -t[0], -t[1]) )
        
        candidates = []
        
        # Generate candidates from sorted_x
        for i in range(len(sorted_x)-1):
            a = sorted_x[i]
            b = sorted_x[i+1]
            val = max(a[0]+b[0], a[1]+b[1], a[2]+b[2])
            candidates.append( (val, a[3], b[3]) )
        
        # Generate candidates from sorted_y
        for i in range(len(sorted_y)-1):
            a = sorted_y[i]
            b = sorted_y[i+1]
            val = max(a[0]+b[0], a[1]+b[1], a[2]+b[2])
            candidates.append( (val, a[3], b[3]) )
        
        # Generate candidates from sorted_z
        for i in range(len(sorted_z)-1):
            a = sorted_z[i]
            b = sorted_z[i+1]
            val = max(a[0]+b[0], a[1]+b[1], a[2]+b[2])
            candidates.append( (val, a[3], b[3]) )
        
        # Sort candidates in descending order of value
        candidates.sort(reverse=True)
        
        # Greedily select the pairs
        used = [False] * N
        total = 0
        selected = 0
        for (val, a, b) in candidates:
            if not used[a] and not used[b]:
                used[a] = True
                used[b] = True
                total += val
                selected += 1
                if selected == K:
                    break
        
        print(total)

if __name__ == "__main__":
    main()