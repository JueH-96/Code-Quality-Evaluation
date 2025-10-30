def find_sick_oyster(S_1, S_2):
    if S_1 == "sick" and S_2 == "fine":
        return 2  # Takahashi got sick, Aoki fine -> oyster 2 is the culprit
    elif S_1 == "fine" and S_2 == "sick":
        return 3  # Takahashi fine, Aoki sick -> oyster 3 is the culprit
    elif S_1 == "fine" and S_2 == "fine":
        return 4  # Both fine -> oyster 4 is the culprit
    elif S_1 == "sick" and S_2 == "sick":
        return 1  # Both sick -> oyster 1 is the culprit

import sys
input = sys.stdin.read

def main():
    S_1, S_2 = input().strip().split()
    result = find_sick_oyster(S_1, S_2)
    print(result)

if __name__ == "__main__":
    main()