import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    n = int(data[0])
    M = int(data[1])
    buckets = [[] for _ in range(M + 2)]
    
    index = 2
    for _ in range(n):
        L = int(data[index])
        R = int(data[index + 1])
        index += 2
        if L <= M:
            buckets[L].append(R)
    
    current_min = M + 1
    f = [0] * (M + 2)
    
    for l in range(M, 0, -1):
        for R_val in buckets[l]:
            if R_val < current_min:
                current_min = R_val
        f[l] = current_min
    
    total_safe = 0
    for l in range(1, M + 1):
        max_r = min(f[l] - 1, M)
        if l <= max_r:
            total_safe += (max_r - l + 1)
    
    print(total_safe)

if __name__ == "__main__":
    main()