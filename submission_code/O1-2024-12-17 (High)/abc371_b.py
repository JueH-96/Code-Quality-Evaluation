def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    
    # This array will track if a family has already had a male baby (Taro assigned).
    taro_assigned = [False] * (N + 1)
    
    # Process each baby's birth information
    idx = 2
    for _ in range(M):
        family = int(data[idx])
        gender = data[idx + 1]
        idx += 2
        
        # If this baby is male and the family has not had a male yet, this baby is Taro.
        if gender == 'M' and not taro_assigned[family]:
            print("Yes")
            taro_assigned[family] = True
        else:
            print("No")

# Do not forget to call the main function
if __name__ == "__main__":
    main()