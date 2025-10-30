import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()

    # c0, c1: counts of substrings ending at previous position
    # whose NAND-accumulation value is 0 or 1 respectively.
    c0 = 0
    c1 = 0
    ans = 0

    for ch in S:
        a = 1 if ch == '1' else 0

        # substrings extended from previous j-1 to j:
        # new_c0 = number of previous substrings with f=1 and a=1 => c1 if a==1 else 0
        if a == 1:
            new_c0 = c1
        else:
            new_c0 = 0
        # new_c1 = all previous substrings minus those that became 0
        new_c1 = c0 + c1 - new_c0

        # add the substring of length 1 starting here:
        if a == 0:
            new_c0 += 1
        else:
            new_c1 += 1

        # substrings ending here that contribute 1 add to answer
        ans += new_c1

        # update for next iteration
        c0, c1 = new_c0, new_c1

    print(ans)

if __name__ == "__main__":
    main()