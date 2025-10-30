def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    S1, S2 = input_data[0], input_data[1]
    
    # Based on eating pattern:
    # Takahashi eats oysters 1 and 2.
    # Aoki eats oysters 1 and 3.
    # The culprit oyster is:
    # 1 if both got sick (because oyster 1 is common),
    # 2 if only Takahashi got sick,
    # 3 if only Aoki got sick,
    # 4 if nobody gets sick.
    
    if S1 == "sick" and S2 == "sick":
        print(1)
    elif S1 == "sick" and S2 == "fine":
        print(2)
    elif S1 == "fine" and S2 == "sick":
        print(3)
    else:
        print(4)

if __name__ == '__main__':
    main()