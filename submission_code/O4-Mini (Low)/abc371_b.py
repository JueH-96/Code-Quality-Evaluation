def main():
    import sys
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    seen_eldest_male = [False] * (N + 1)
    
    for _ in range(M):
        fam, gender = input().split()
        fam = int(fam)
        if gender == 'M' and not seen_eldest_male[fam]:
            print("Yes")
            seen_eldest_male[fam] = True
        else:
            print("No")

if __name__ == "__main__":
    main()