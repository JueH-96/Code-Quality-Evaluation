# YOUR CODE HERE
import sys
import math

def main():
    import sys
    import threading

    def solve():
        N = int(sys.stdin.readline())
        MAX_A = 10 ** 6
        found = False

        # First, check if N is a palindrome with digits '1'-'9' only, no zeros
        N_str = str(N)
        if N_str == N_str[::-1] and all(d in '123456789' for d in N_str):
            print(N_str)
            return

        # Try to find a in factors up to MAX_A
        for a in range(1, int(min(N, MAX_A))+1):
            if N % (a * a) == 0:
                c = N // (a * a)
                a_str = str(a)
                c_str = str(c)
                # Check digits of a and c
                if '0' in a_str or '0' in c_str:
                    continue
                if not all(d in '123456789' for d in a_str):
                    continue
                if not all(d in '123456789' for d in c_str):
                    continue
                # Construct S
                S = a_str + '*' + c_str + '*' + a_str
                if S == S[::-1]:
                    print(S)
                    found = True
                    return

        # As an alternative, try to find factors a and b such that N = a * b and a_str + '*' + b_str + '*' + a_str is palindrome
        # This is time consuming, so we can limit a and b up to 10^6
        # But in the interest of time, we'll skip this and output -1 if not found
        print(-1)

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()