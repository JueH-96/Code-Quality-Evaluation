import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    M = int(data[1])
    events = [[] for _ in range(M + 1)]
    index = 2
    for _ in range(n):
        L_i = int(data[index])
        R_i = int(data[index + 1])
        index += 2
        if L_i <= M:
            events[L_i].append(R_i)
    
    min_val = M + 1
    total_union = 0
    for l in range(M, 0, -1):
        for r_val in events[l]:
            if r_val < min_val:
                min_val = r_val
        if min_val < M + 1:
            total_union += (M - min_val + 1)
    
    total_pairs = M * (M + 1) // 2
    ans = total_pairs - total_union
    print(ans)

if __name__ == '__main__':
    main()