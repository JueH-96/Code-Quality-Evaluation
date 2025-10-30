def is_airport_code(S, T):
    """
    Checks if T is an airport code for S.

    Args:
    S (str): The string of lowercase English letters.
    T (str): The string of uppercase English letters.

    Returns:
    bool: True if T is an airport code for S, False otherwise.
    """
    # Check if T can be derived from S by taking a subsequence of length 3
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            for k in range(j + 1, len(S)):
                if S[i].upper() == T[0] and S[j].upper() == T[1] and S[k].upper() == T[2]:
                    return True

    # Check if T can be derived from S by taking a subsequence of length 2 and appending X
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if S[i].upper() == T[0] and S[j].upper() == T[1] and T[2] == 'X':
                return True

    return False


def main():
    S = input()
    T = input()

    if is_airport_code(S, T):
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()