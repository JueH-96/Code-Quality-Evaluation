def main():
    N, M = map(int, input().split())
    eldest_sons = set()
    for _ in range(M):
        A_i, B_i = input().split()
        if B_i == 'M' and A_i not in eldest_sons:
            print("Yes")
            eldest_sons.add(A_i)
        else:
            print("No")

if __name__ == "__main__":
    main()