def solve():
    n = int(input())
    chords = []
    for _ in range(n):
        u, v = map(int, input().split())
        chords.append(tuple(sorted((u, v))))
    
    has_intersection = False
    for i in range(n):
        for j in range(i + 1, n):
            u1, v1 = chords[i]
            u2, v2 = chords[j]
            condition1 = (u1 < u2 < v1 < v2)
            condition2 = (u2 < u1 < v2 < v1)
            if condition1 or condition2:
                has_intersection = True
                break
        if has_intersection:
            break
            
    if has_intersection:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()