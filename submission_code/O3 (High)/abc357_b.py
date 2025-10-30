import sys

def main() -> None:
    S = sys.stdin.readline().rstrip()

    upper_cnt = sum(1 for ch in S if ch.isupper())
    lower_cnt = len(S) - upper_cnt  # since S contains only letters

    if upper_cnt > lower_cnt:
        # More uppercase letters – turn all into uppercase
        result = S.upper()
    else:
        # Otherwise – turn all into lowercase
        result = S.lower()

    print(result)

if __name__ == "__main__":
    main()