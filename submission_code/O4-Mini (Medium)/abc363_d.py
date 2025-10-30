import sys
import threading

def main():
    import sys

    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)

    # The 1st through 10th palindromes are 0,1,2,...,9
    if N <= 10:
        # 1->0, 2->1, ..., 10->9
        print(N-1)
        return

    # Remove the first 10
    N -= 10

    # Now consider palindromes of length L>=2
    L = 2
    while True:
        # half = number of digits in the "prefix" that determines the palindrome
        half = (L + 1) // 2
        # count of palindromes of length L:
        # - first digit can't be zero, so 9 choices for first digit,
        # - the remaining (half-1) digits of the prefix free in [0..9]
        cnt = 9 * pow(10, half - 1)
        if N > cnt:
            N -= cnt
            L += 1
        else:
            # the N-th palindrome of length L
            # offset in [0..cnt-1]
            offset = N - 1
            # the first half number:
            first_half_num = pow(10, half - 1) + offset
            s = str(first_half_num)
            if L % 2 == 0:
                # even length: mirror entire s
                pal = s + s[::-1]
            else:
                # odd length: mirror all but the last digit of s
                pal = s + s[:-1][::-1]
            print(pal)
            return

if __name__ == "__main__":
    main()