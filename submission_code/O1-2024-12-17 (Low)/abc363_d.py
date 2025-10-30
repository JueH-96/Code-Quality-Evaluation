def main():
    import sys
    sys.setrecursionlimit(10**7)

    N = int(sys.stdin.readline().strip())

    # Counts how many palindromes have exactly L digits.
    def palindrome_count_exact(L):
        if L == 1:
            return 10  # [0..9]
        if L % 2 == 0:
            # Even length L = 2k -> determined by first k digits (first digit != 0)
            return 9 * pow(10, (L // 2) - 1)
        else:
            # Odd length L = 2k+1 -> determined by first k+1 digits (first digit != 0)
            return 9 * pow(10, (L - 1) // 2)

    # Find the Nth smallest palindrome
    # We'll accumulate counts by length until we reach N
    def nth_palindrome(N):
        # We iterate over possible lengths until sum of counts >= N
        length = 1
        cumulative = 0
        while True:
            cnt = palindrome_count_exact(length)
            if cumulative + cnt >= N:
                # Nth palindrome has 'length' digits (special case length=1)
                offset = N - cumulative - 1  # 0-based offset within length-digit palindromes
                if length == 1:
                    # offset=0 -> 0, offset=1 -> 1, ..., offset=9 -> 9
                    return str(offset)
                else:
                    # Construct the palindrome from its prefix
                    half_len = (length + 1) // 2
                    prefix_num = pow(10, half_len - 1) + offset
                    prefix_str = str(prefix_num)
                    if length % 2 == 0:
                        # Even length => mirror entire prefix
                        return prefix_str + prefix_str[::-1]
                    else:
                        # Odd length => mirror all but last char of prefix
                        return prefix_str + prefix_str[-2::-1]
            cumulative += cnt
            length += 1

    # Compute and print the result
    print(nth_palindrome(N))

# Call main() to run the solution
main()