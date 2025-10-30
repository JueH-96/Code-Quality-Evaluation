# YOUR CODE HERE
def main():
    import math, sys
    sys.setrecursionlimit(10**6)
    
    data = sys.stdin.read().strip().split()
    if not data:
        return
    try:
        N = int(data[0])
    except:
        return
    
    # helper: check if a string is palindrome and has no '0'
    def valid_str(s):
        return ('0' not in s) and (s == s[::-1])
    
    # --- Option 1: try S = str(N) if it's valid ---
    sN = str(N)
    if valid_str(sN):
        print(sN)
        return
    
    # --- Option 2: try a representation of the form A * B * A with A, B palindromic ---
    # We want A^2 * B = N. We search for candidate A such that A^2 divides N.
    # We only consider A that are palindromic and have no '0'
    # A's value: note that A^2 <= N, so A <= sqrt(N)
    maxA = int(math.isqrt(N))
    found = False
    for A in range(1, maxA+1):
        sA = str(A)
        if '0' in sA or sA != sA[::-1]:
            continue
        if N % (A*A) != 0:
            continue
        B = N // (A*A)
        sB = str(B)
        if '0' in sB or sB != sB[::-1]:
            continue
        S = sA + "*" + sB + "*" + sA
        # check length condition
        if 1 <= len(S) <= 1000:
            print(S)
            return

    # --- Option 3: try the 2–factor approach: representation S = A * B  ---
    # We require N = A * B and also that str(A) is the reverse of str(B) so that
    # S = str(A) + "*" + str(B) is palindromic.
    # We iterate over divisors A up to sqrt(N)
    maxA2 = int(math.isqrt(N))
    for A in range(1, maxA2+1):
        if N % A != 0:
            continue
        B = N // A
        sA = str(A)
        sB = str(B)
        if ('0' in sA) or ('0' in sB):
            continue
        # Palindrome condition for S: we need sA + "*" + sB equal to (sA + "*" + sB)[::-1]
        # That forces sA == sB[::-1].
        if sA == sB[::-1]:
            S = sA + "*" + sB
            if 1 <= len(S) <= 1000:
                print(S)
                return

    # If none of the above methods worked, output -1.
    print(-1)
    
if __name__ == '__main__':
    main()