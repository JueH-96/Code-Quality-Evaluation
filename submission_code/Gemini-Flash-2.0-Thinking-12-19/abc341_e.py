def solve():
    n, q = map(int, input().split())
    s_list = list(input())
    
    for _ in range(q):
        query = list(map(int, input().split()))
        query_type = query[0]
        l, r = query[1], query[2]
        
        if query_type == 1:
            for i in range(l - 1, r):
                if s_list[i] == '0':
                    s_list[i] = '1'
                else:
                    s_list[i] = '0'
        elif query_type == 2:
            substring = s_list[l-1:r]
            is_good = True
            if len(substring) > 1:
                for i in range(1, len(substring)):
                    if substring[i] == substring[i-1]:
                        is_good = False
                        break
            if is_good:
                print("Yes")
            else:
                print("No")

if __name__ == '__main__':
    solve()