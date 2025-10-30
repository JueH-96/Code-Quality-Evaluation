def solve():
    n, m = map(int, input().split())
    first_male_born = [0] * (n + 1)
    for _ in range(m):
        family_id, gender = input().split()
        family_id = int(family_id)
        if gender == 'F':
            print("No")
        elif gender == 'M':
            if first_male_born[family_id] == 0:
                print("Yes")
                first_male_born[family_id] = 1
            else:
                print("No")

if __name__ == '__main__':
    solve()