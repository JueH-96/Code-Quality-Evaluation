import sys
import threading

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # K is given but fixed to 1 for this subproblem
    # K = int(data[0])
    S = data[1].rstrip('
')
    T = data[2].rstrip('
')
    len_s = len(S)
    len_t = len(T)
    # If strings are already equal, zero operations needed
    if S == T:
        print("Yes")
        return
    # If length difference more than 1, cannot do in one edit
    if abs(len_s - len_t) > 1:
        print("No")
        return
    # Case 1: lengths equal -> check for at most one substitution
    if len_s == len_t:
        diff = 0
        for cs, ct in zip(S, T):
            if cs != ct:
                diff += 1
                if diff > 1:
                    print("No")
                    return
        print("Yes")
        return

    # Case 2 and 3: lengths differ by exactly 1 -> check one insert/delete
    # Let 'shorter' be the shorter string, 'longer' the longer
    if len_s < len_t:
        shorter, longer = S, T
    else:
        shorter, longer = T, S

    i = j = 0
    skipped = False
    len_short = len(shorter)
    len_long = len(longer)
    while i < len_short and j < len_long:
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            if skipped:
                print("No")
                return
            skipped = True
            j += 1  # skip one char in the longer string
    # If we finish loop without more than one skip, it's valid
    print("Yes")

if __name__ == "__main__":
    main()