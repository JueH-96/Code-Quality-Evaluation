def solve():
    n = int(input())
    current_num = n
    while True:
        s_num = str(current_num)
        if len(s_num) == 3:
            hundreds = int(s_num[0])
            tens = int(s_num[1])
            ones = int(s_num[2])
            if hundreds * tens == ones:
                print(current_num)
                return
        current_num += 1

solve()