def main():
    import sys
    data = sys.stdin.read().split()
    if len(data) < 2:
        return
    S1, S2 = data[0], data[1]
    
    # Evaluate based on the given conditions:
    # Takahashi ate oysters 1 and 2.
    # Aoki ate oysters 1 and 3.
    
    if S1 == "sick" and S2 == "sick":
        # Both got sick, common oyster is 1.
        print(1)
    elif S1 == "sick" and S2 == "fine":
        # Takahashi got sick, so dangerous must be 1 or 2;
        # Aoki didn't get sick, so oyster 1 is not dangerous.
        print(2)
    elif S1 == "fine" and S2 == "sick":
        # Aoki got sick, so dangerous must be 1 or 3;
        # Takahashi didn't get sick, so oyster 1 is not dangerous.
        print(3)
    elif S1 == "fine" and S2 == "fine":
        # Neither got sick, hence dangerous oyster must be 4.
        print(4)

if __name__ == '__main__':
    main()