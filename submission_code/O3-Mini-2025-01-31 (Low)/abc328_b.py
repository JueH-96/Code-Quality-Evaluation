def is_repdigit(x: int) -> bool:
    s = str(x)
    return all(ch == s[0] for ch in s)

def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    # D[i] is the number of days in month i+1.
    days_per_month = list(map(int, input().split()))
    
    ans = 0
    # Iterate over each month 1-indexed.
    for i in range(1, N + 1):
        # Only consider months that are repdigit.
        if is_repdigit(i):
            # extract the repeating digit used in the month
            dchar = str(i)[0]
            # For the day to have repdigit date with the month, 
            # the day must be a repdigit number that is constructed solely from dchar.
            # Since D_i (number of days in month) is at most 100, we only need to try lengths 1 to 3.
            for length in range(1, 4):
                rep_day = int(dchar * length)
                if rep_day <= days_per_month[i-1]:
                    ans += 1
    print(ans)

if __name__ == '__main__':
    main()