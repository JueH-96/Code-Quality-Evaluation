def solve():
    R, B = map(int, input().split())
    total = R + B
    
    if total == 2 and (R == 1 and B == 1):
        print("No")
        return
    
    print("Yes")
    
    if R >= total // 2:
        for i in range(R):
            print("R", 1, i + 1)
        for i in range(B):
            print("B", 2, i + 1)
    else:
        for i in range(B):
            print("B", 1, i + 1)
        for i in range(R):
            print("R", 2, i + 1)

T = int(input())
for _ in range(T):
    solve()