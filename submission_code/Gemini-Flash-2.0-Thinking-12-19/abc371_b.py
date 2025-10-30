def solve():
    n, m = map(int, input().split())
    first_male_born = [False] * (n + 1)
    for _ in range(m):
        family_index, gender = input().split()
        family_index = int(family_index)
        if gender == 'F':
            print("No")
        elif gender == 'M':
            if not first_male_born[family_index]:
                print("Yes")
                first_male_born[family_index] = True
            else:
                print("No")

if __name__ == '__main__':
    solve()