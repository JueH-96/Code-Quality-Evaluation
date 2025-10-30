# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    st = []
    for _ in range(n - 1):
        st.append(list(map(int, input().split())))

    ans = a[n-1]
    for i in range(1 << (n - 1)):
        temp_a = list(a)
        curr_ans = temp_a[n-1]
        for j in range(n - 1):
            if (i >> j) & 1:
                count = temp_a[j] // st[j][0]
                temp_a[j] -= count * st[j][0]
                temp_a[j+1] += count * st[j][1]
        
        curr_ans = temp_a[n-1]
        ans = max(ans, curr_ans)

    print(ans)

solve()