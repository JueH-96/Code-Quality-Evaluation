def main():
    import sys
    input = sys.stdin.readline

    # Read number of families and number of babies
    N, M = map(int, input().split())
    
    # Array to record whether a male baby has been born in each family.
    # Using index 1..N (so we create list with N+1 elements)
    male_born = [False] * (N + 1)
    
    # Process each baby's record in chronological order.
    for _ in range(M):
        # Read family id and gender
        data = input().split()
        family_id = int(data[0])
        gender = data[1]
        
        # Determine if the baby gets named Taro.
        # Taro is only given to the earliest born male in the family.
        if gender == 'M' and not male_born[family_id]:
            print("Yes")
            male_born[family_id] = True
        else:
            print("No")

if __name__ == '__main__':
    main()