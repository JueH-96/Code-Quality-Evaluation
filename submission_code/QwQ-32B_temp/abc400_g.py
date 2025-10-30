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
        cakes = []
        for i in range(N):
            X = int(input[ptr])
            Y = int(input[ptr+1])
            Z = int(input[ptr+2])
            cakes.append((X, Y, Z))
            ptr += 3
        candidates = []
        for criteria in range(3):
            sorted_indices = sorted(range(N), key=lambda i: -cakes[i][criteria])
            sub = sorted_indices[:2*K]
            for i in range(0, len(sub), 2):
                if i + 1 >= len(sub):
                    break
                a = sub[i]
                b = sub[i+1]
                xsum = cakes[a][0] + cakes[b][0]
                ysum = cakes[a][1] + cakes[b][1]
                zsum = cakes[a][2] + cakes[b][2]
                val = max(xsum, ysum, zsum)
                candidates.append((val, a, b))
        # Sort candidates in descending order of value
        candidates.sort(key=lambda x: -x[0])
        used = [False] * N
        total = 0
        count = 0
        for cand in candidates:
            val, a, b = cand
            if not used[a] and not used[b]:
                used[a] = True
                used[b] = True
                total += val
                count += 1
                if count == K:
                    break
        print(total)

if __name__ == "__main__":
    main()