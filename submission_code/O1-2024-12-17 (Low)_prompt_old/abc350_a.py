def solve():
    S = input().strip()
    # Since S is guaranteed to start with 'ABC' and end with digits, extract the numeric part
    num_part = int(S[3:])
    # Check the conditions:
    # - 1 <= num_part <= 315 or 317 <= num_part <= 349
    # - Excluding 316
    if (1 <= num_part <= 315) or (317 <= num_part <= 349):
        print("Yes")
    else:
        print("No")

def main():
    solve()

if __name__ == "__main__":
    main()