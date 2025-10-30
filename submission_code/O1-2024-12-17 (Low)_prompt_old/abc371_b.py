def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])

    # Track if a family has already had a male child
    has_male_child = [False] * (N + 1)  # 1-based indexing

    index = 2
    for _ in range(M):
        family = int(data[index])
        gender = data[index + 1]
        index += 2

        if gender == 'M' and not has_male_child[family]:
            print("Yes")
            has_male_child[family] = True
        else:
            print("No")

def main():
    solve()

if __name__ == "__main__":
    main()