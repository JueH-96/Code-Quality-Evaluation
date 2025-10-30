def solve():
    n, q = map(int, input().split())
    s_str = input()
    s_list = list(s_str)
    
    for _ in range(q):
        x, c = input().split()
        x = int(x)
        s_list[x-1] = c
        current_s = "".join(s_list)
        count = 0
        for i in range(len(current_s) - 2):
            if current_s[i:i+3] == "ABC":
                count += 1
        print(count)

if __name__ == '__main__':
    solve()