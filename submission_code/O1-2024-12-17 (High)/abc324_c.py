import sys

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    T_prime = data[1]
    answers = []

    # Helper function: check if two strings of the same length differ by exactly one character
    def differ_by_one_char(a, b):
        mismatch = 0
        for x, y in zip(a, b):
            if x != y:
                mismatch += 1
                if mismatch > 1:
                    return False
        return mismatch == 1

    # Helper function: check if t is formed by inserting exactly one character into s
    # i.e. len(t) == len(s) + 1
    def is_insertion_one(s, t):
        ls, lt = len(s), len(t)
        i = j = 0
        inserted = False
        while i < ls and j < lt:
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                if inserted:
                    return False
                inserted = True
                j += 1
        # If we haven't used insertion yet, consume the last leftover char in t
        if not inserted:
            inserted = True
            j += 1
        # Check that we used up s entirely, t entirely, and used exactly one insertion
        return (i == ls and j == lt and inserted)

    # The core check function:
    # Return True if T_prime can be derived from s by exactly one of the four conditions:
    # 1) T' == s
    # 2) T' is s + one inserted char
    # 3) T' is s - one deleted char
    # 4) T' differs from s by exactly one replaced char (same length)
    def can_derive_tprime(s, tprime):
        ls, lt = len(s), len(tprime)
        # Condition 1 or 4 (same length)
        if ls == lt:
            # 1) Exactly the same
            if s == tprime:
                return True
            # 4) Differ by one character
            return differ_by_one_char(s, tprime)
        # Condition 2 (insert one char): tprime has length one more than s
        elif ls + 1 == lt:
            return is_insertion_one(s, tprime)
        # Condition 3 (delete one char): tprime has length one less than s
        elif ls - 1 == lt:
            # Same as "tprime is formed by inserting one char into tprime to get s"
            # or just check "s => tprime by deleting one char" via the same logic
            return is_insertion_one(tprime, s)
        else:
            return False

    idx = 2
    for i in range(1, N + 1):
        candidate = data[idx]
        idx += 1
        if can_derive_tprime(candidate, T_prime):
            answers.append(i)

    print(len(answers))
    if answers:
        print(" ".join(map(str, answers)))

# Do not forget to call main()!
if __name__ == "__main__":
    main()