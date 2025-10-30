def solve():
    n, q = map(int, input().split())
    s_list = list(input())
    
    def is_good_string(sub_s_list):
        if len(sub_s_list) <= 1:
            return True
        for i in range(len(sub_s_list) - 1):
            if sub_s_list[i] == sub_s_list[i+1]:
                return False
        return True
        
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
            substring_list = s_list[l-1:r]
            if is_good_string(substring_list):
                print("Yes")
            else:
                print("No")
                
if __name__ == '__main__':
    solve()