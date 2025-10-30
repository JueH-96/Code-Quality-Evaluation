def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    # The first 10 palindromes are 0,1,...,9
    if N <= 10:
        print(N - 1)
        return

    # Subtract out the single‐digit palindromes
    N -= 10

    # Now consider palindromes of length L = 2,3,4,...
    L = 2
    while True:
        # half‐length (number of digits to choose) 
        h = (L + 1) // 2
        # count of palindromes of length L:
        # first half has 9 * 10^(h-1) choices (first digit ≠ 0)
        cnt = 9 * (10 ** (h - 1))
        if N <= cnt:
            # the N-th palindrome of length L
            # prefix runs from 10^(h-1) to 10^h - 1
            start = 10 ** (h - 1)
            prefix = start + (N - 1)
            s = str(prefix)
            if L % 2 == 0:
                # even length: mirror entire prefix
                pal = s + s[::-1]
            else:
                # odd length: mirror all but the last digit of prefix
                pal = s + s[:-1][::-1]
            print(pal)
            return
        # otherwise skip these and go to next length
        N -= cnt
        L += 1

# call main
main()