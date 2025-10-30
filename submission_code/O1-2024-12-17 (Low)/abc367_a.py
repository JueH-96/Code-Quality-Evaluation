def main():
    A, B, C = map(int, input().split())
    
    # If B < C, Takahashi sleeps between B and C (including B, excluding C)
    # If B > C, Takahashi sleeps from B to 24 and 0 to C.
    
    if B < C:
        # Check if A is within sleeping time
        if B <= A < C:
            print("No")
        else:
            print("Yes")
    else:
        # B > C
        # Sleeping time is [B, 24) and [0, C)
        # A is awake if it's not in those intervals
        # So if A >= B or A < C, he is asleep
        if A >= B or A < C:
            print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    main()