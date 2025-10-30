def main():
    import sys
    S = sys.stdin.readline().strip()

    i = 0
    presses = 0
    n = len(S)

    while i < n:
        if S[i] != '0':
            # For any digit 1..9, press once
            presses += 1
            i += 1
        else:
            # Count how many consecutive zeros
            start = i
            while i < n and S[i] == '0':
                i += 1
            length_of_zeros = i - start
            # Each pair of zeros (00) can be pressed with one button
            # So number of presses for k consecutive zeros is k//2 + k%2
            presses += (length_of_zeros // 2) + (length_of_zeros % 2)

    print(presses)

# Do not forget to call main() at the end
main()