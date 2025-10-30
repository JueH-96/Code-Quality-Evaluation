# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    t = int(data[0])
    # helper function: compute minimal period length of s.
    def min_period(s):
        n = len(s)
        pi = [0] * n
        j = 0
        for i in range(1, n):
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        period = n - pi[-1]
        if n % period == 0:
            return period
        else:
            return n

    out_lines = []
    line_index = 1
    for _ in range(t):
        s = data[line_index].rstrip('
')
        line_index += 1
        x = data[line_index].rstrip('
')
        line_index += 1
        y = data[line_index].rstrip('
')
        line_index += 1

        n_s = len(s)
        a0 = x.count('0')
        a1 = len(x) - a0
        b0 = y.count('0')
        b1 = len(y) - b0

        # Case 1: if number of ones are equal, then also the zeros must match.
        if a1 == b1:
            if a0 == b0:
                out_lines.append("Yes")
            else:
                out_lines.append("No")
        else:
            # a1 != b1. The length condition gives:
            #   (a1-b1)*L = (b0-a0)*|S|
            # so candidate L is:
            diff = a1 - b1
            num = (b0 - a0) * n_s
            if diff == 0 or num % diff != 0:
                out_lines.append("No")
            else:
                L = num // diff
                if L < 0:
                    out_lines.append("No")
                elif L == 0:
                    # T empty works.
                    out_lines.append("Yes")
                else:
                    # When T is nonempty, S and T must commute.
                    # In that case T must be a power of the primitive word u that S also is a power of.
                    # In particular, if d is the minimal period length of s, then L must be a multiple of d.
                    d = min_period(s)
                    if L % d == 0:
                        out_lines.append("Yes")
                    else:
                        out_lines.append("No")
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()