def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    A, B, C = map(int, input_data)
    
    # When B < C, Takahashi is asleep from B to C.
    # When B > C, his sleeping time spans over midnight: from B to 23 and from 0 to C.
    # Determine if A is during his sleep.
    if B < C:
        asleep = (B <= A < C)
    else:
        asleep = (A >= B or A < C)
    
    print("No" if asleep else "Yes")

if __name__ == '__main__':
    main()