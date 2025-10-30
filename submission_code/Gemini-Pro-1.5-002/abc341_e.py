# YOUR CODE HERE
def solve():
    n, q = map(int, input().split())
    s = list(input())
    
    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            l, r = query[1], query[2]
            for i in range(l - 1, r):
                s[i] = '1' if s[i] == '0' else '0'
        else:
            l, r = query[1], query[2]
            sub_s = s[l-1:r]
            is_good = True
            for i in range(len(sub_s) - 1):
                if sub_s[i] == sub_s[i+1]:
                    is_good = False
                    break
            print("Yes" if is_good else "No")

solve()