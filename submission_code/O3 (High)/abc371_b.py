import sys

def main() -> None:
    input = sys.stdin.readline

    N, M = map(int, input().split())
    # has_male[i] == True means family i has already had a boy
    has_male = [False] * (N + 1)

    for _ in range(M):
        family_str, gender = input().split()
        family = int(family_str)

        if gender == 'M' and not has_male[family]:
            print("Yes")
            has_male[family] = True   # first boy in this family
        else:
            print("No")

if __name__ == "__main__":
    main()