def solve():
    import sys
    data = sys.stdin.read().strip().split()
    A, B, C = map(int, data)

    # Takahashi sleeps from B (inclusive) to C (exclusive) in a 24-hour cycle.
    # We check if A (the time to shout) falls within the awake period.
    
    # If B < C, sleeping interval is [B, C).
    # Otherwise, sleeping interval is [B, 24) and [0, C).
    
    if B < C:
        # If A is between B (inclusive) and C (exclusive), he is asleep
        if B <= A < C:
            print("No")
        else:
            print("Yes")
    else:
        # B > C
        # Sleeping intervals: [B, 24) and [0, C)
        # If A >= B or A < C, then A is in the sleeping interval
        if A >= B or A < C:
            print("No")
        else:
            print("Yes")

def main():
    solve()

if __name__ == "__main__":
    main()