def main():
    import sys
    from collections import Counter

    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    allowed_letters = set(['a', 't', 'c', 'o', 'd', 'e', 'r'])

    # Count frequencies in S and T, excluding '@'
    freq_S = Counter(c for c in S if c != '@')
    freq_T = Counter(c for c in T if c != '@')

    # Count number of '@' in S and T
    num_at_S = S.count('@')
    num_at_T = T.count('@')

    # Check for letters not in allowed_letters
    for c in set(freq_S.keys()).union(set(freq_T.keys())):
        if c not in allowed_letters:
            if freq_S.get(c, 0) != freq_T.get(c, 0):
                print("No")
                return

    # Calculate additions needed for allowed letters
    additions_S = 0
    additions_T = 0
    for c in allowed_letters:
        freq_S_c = freq_S.get(c, 0)
        freq_T_c = freq_T.get(c, 0)
        if freq_S_c < freq_T_c:
            additions_S += freq_T_c - freq_S_c
        elif freq_S_c > freq_T_c:
            additions_T += freq_S_c - freq_T_c

    # Check if additions do not exceed the number of '@' symbols
    if additions_S <= num_at_T and additions_T <= num_at_S:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()