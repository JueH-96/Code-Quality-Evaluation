def main():
    import sys
    from itertools import permutations

    # Read the input relationships S_AB, S_AC, S_BC
    S_AB, S_AC, S_BC = sys.stdin.read().split()

    # We'll define all permutations of A, B, C in ascending age order
    # (the first in the tuple is the youngest, the third is the oldest)
    all_perms = permutations(["A", "B", "C"])

    def satisfies_constraints(perm, sab, sac, sbc):
        # perm is a tuple like ('A','B','C'), meaning A is youngest, B is middle, C is oldest
        # index_of[X] gives the rank of X (0 for youngest, 1 for middle, 2 for oldest)
        index_of = {p: i for i, p in enumerate(perm)}

        # Check S_AB
        if sab == "<":
            if not (index_of["A"] < index_of["B"]):
                return False
        else:  # sab == ">"
            if not (index_of["A"] > index_of["B"]):
                return False

        # Check S_AC
        if sac == "<":
            if not (index_of["A"] < index_of["C"]):
                return False
        else:  # sac == ">"
            if not (index_of["A"] > index_of["C"]):
                return False

        # Check S_BC
        if sbc == "<":
            if not (index_of["B"] < index_of["C"]):
                return False
        else:  # sbc == ">"
            if not (index_of["B"] > index_of["C"]):
                return False

        return True

    # Find the permutation that satisfies the constraints
    for p in all_perms:
        if satisfies_constraints(p, S_AB, S_AC, S_BC):
            # The middle brother is at index 1
            print(p[1])
            break

# Do not forget to call main() at the end.
if __name__ == "__main__":
    main()