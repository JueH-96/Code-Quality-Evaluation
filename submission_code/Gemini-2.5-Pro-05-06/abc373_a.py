def solve():
    count = 0
    for i in range(1, 13):  # i will take values from 1 to 12
        s_i = input()
        if len(s_i) == i:
            count += 1
    print(count)

if __name__ == '__main__':
    solve()