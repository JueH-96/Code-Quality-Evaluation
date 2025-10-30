def main():
    import sys
    input_data = sys.stdin.read().split()
    A, B, C = map(int, input_data)
    
    # If wake up time C is greater than or equal to sleep time B,
    # then sleeping period is from B to C (non-inclusive equals awake).
    if B < C:
        # If A is during sleep time [B, C), then "No".
        if B <= A < C:
            print("No")
        else:
            print("Yes")
    
    # If B > C, the sleeping period wraps past midnight.
    else:
        # The asleep intervals are [B, 24) and [0, C)
        if A >= B or A < C:
            print("No")
        else:
            print("Yes")

if __name__ == '__main__':
    main()