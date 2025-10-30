# YOUR CODE HERE
def solve():
    n = int(input())
    d = list(map(int, input().split()))
    
    ans = 0
    for i in range(1, n + 1):
        s_i = str(i)
        if len(set(s_i)) == 1:
            rep_digit_i = int(s_i[0])
            for j in range(1, d[i-1] + 1):
                s_j = str(j)
                if len(set(s_j)) == 1 and int(s_j[0]) == rep_digit_i:
                    ans += 1
    print(ans)

solve()