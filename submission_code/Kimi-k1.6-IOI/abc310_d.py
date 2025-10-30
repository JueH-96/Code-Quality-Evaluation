import math

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    incompat = [[] for _ in range(N)]
    for _ in range(M):
        a = int(input[ptr]) - 1
        ptr += 1
        b = int(input[ptr]) - 1
        ptr += 1
        incompat[a].append(b)
        incompat[b].append(a)
    
    count = 0
    colors = [-1] * N
    
    def backtrack(pos, used):
        nonlocal count
        if pos == N:
            if bin(used).count('1') == T:
                count += 1
            return
        for color in range(T):
            valid = True
            for j in incompat[pos]:
                if j < pos and colors[j] == color:
                    valid = False
                    break
            if not valid:
                continue
            current_used = bin(used | (1 << color)).count('1')
            remaining = N - pos - 1
            if current_used + remaining < T:
                continue
            colors[pos] = color
            backtrack(pos + 1, used | (1 << color))
            colors[pos] = -1
    
    backtrack(0, 0)
    fact = math.factorial(T)
    print(count // fact if fact != 0 else 0)

if __name__ == "__main__":
    main()