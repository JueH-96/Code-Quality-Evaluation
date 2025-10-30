# YOUR CODE HERE
def solve():
    import sys
    import sys
    def input():
        return sys.stdin.read()
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    S = set()
    idx = 2
    for _ in range(M):
        a = int(data[idx])
        b = int(data[idx+1])
        S.add(a * 10**10 + b)
        idx +=2
    moves = [ (2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1) ]
    A_list = []
    for encoded in S:
        i = encoded // 10**10
        j = encoded % 10**10
        for dx, dy in moves:
            ni = i + dx
            nj = j + dy
            if 1 <= ni <= N and 1 <= nj <= N:
                A_list.append(ni * 10**10 + nj)
    A = set(A_list)
    overlap = 0
    for x in A:
        if x in S:
            overlap +=1
    attacked_empty = len(A) - overlap
    total_empty = (N * N) - M
    answer = total_empty - attacked_empty
    print(answer)