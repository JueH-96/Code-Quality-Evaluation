def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])

    # KMP prefix-function to help find the length of the minimal period.
    def prefix_function(s: str) -> list:
        pi = [0] * len(s)
        k = 0
        for i in range(1, len(s)):
            while k > 0 and s[k] != s[i]:
                k = pi[k - 1]
            if s[k] == s[i]:
                k += 1
            pi[i] = k
        return pi

    # Returns the length of the smallest period P of s
    # so that s = P^(n/|P|) if possible.
    def minimal_period_length(s: str) -> int:
        n = len(s)
        pi = prefix_function(s)
        border = pi[-1]
        candidate = n - border
        if n % candidate == 0:
            return candidate
        return n

    idx = 1  # pointer in input_data
    answers = []

    for _ in range(t):
        S = input_data[idx]; idx += 1
        X = input_data[idx]; idx += 1
        Y = input_data[idx]; idx += 1

        # If X and Y are exactly the same, we can always produce the same f(S,T,X)=f(S,T,Y).
        if X == Y:
            answers.append("Yes")
            continue

        # Compute the minimal period of S
        n = len(S)
        c = minimal_period_length(S)
        p = n // c  # S = (some base-string) ^ p

        # Count how many zeros/ones in X and Y
        X0 = X.count('0')
        X1 = len(X) - X0
        Y0 = Y.count('0')
        Y1 = len(Y) - Y0

        # A,B for the length equation; we want p*A + q*B = 0 for some q >= 0:
        A = X0 - Y0
        B = X1 - Y1

        # Case: B == 0 implies X1=Y1
        if B == 0:
            # Must have A == 0 too (X0=Y0) or there's no way lengths match
            # when lumps differ somewhere (since X!=Y).
            if A == 0:
                answers.append("Yes")
            else:
                answers.append("No")
        else:
            # Solve for q = -p*A / B, must be integer >=0
            num = -p * A
            if num % B != 0:
                answers.append("No")
            else:
                q = num // B
                if q >= 0:
                    answers.append("Yes")
                else:
                    answers.append("No")

    print("
".join(answers))

# do not forget to call main
if __name__ == "__main__":
    main()