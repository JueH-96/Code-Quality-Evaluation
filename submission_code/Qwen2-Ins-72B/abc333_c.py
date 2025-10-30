def repunit(n):
    return int('1' * n)

def main():
    N = int(input())
    ans_set = set()
    for i in range(1, 12):
        for j in range(i, 12):
            for k in range(j, 12):
                ans_set.add(repunit(i) + repunit(j) + repunit(k))
    ans_list = sorted(list(ans_set))
    print(ans_list[N-1])

if __name__ == '__main__':
    main()