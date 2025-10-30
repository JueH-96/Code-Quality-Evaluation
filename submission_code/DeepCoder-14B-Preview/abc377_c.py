def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    S = set()
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        S.add((a, b))
    
    deltas = [ (2, 1), (1, 2), (-1, 2), (-2, 1),
               (-2, -1), (-1, -2), (1, -2), (2, -1) ]
    
    T = set()
    for (a, b) in S:
        for dx, dy in deltas:
            x = a + dx
            y = b + dy
            if 1 <= x <= N and 1 <= y <= N:
                T.add((x, y))
    
    len_T = len(T)
    intersection = T & S
    len_intersection = len(intersection)
    
    safe = (N * N - M) - (len_T - len_intersection)
    print(safe)

if __name__ == '__main__':
    main()