def solve():
    import sys
    input_data=sys.stdin.read().strip().split()
    t=int(input_data[0])
    # We will parse the input as triples of lines: S, X, Y
    
    # A helper to compute the length of the minimal period of a string
    # using the prefix-function (a.k.a. failure function in KMP).
    def minimal_period_length(s):
        # Compute prefix-function in O(len(s)).
        n = len(s)
        pi = [0]*n
        # Standard prefix‐function (KMP) computation
        j=0
        for i in range(1,n):
            while j>0 and s[i] != s[j]:
                j = pi[j-1]
            if s[i]==s[j]:
                j+=1
            pi[i]=j
        # If n % (n - pi[n-1]) == 0, the minimal period is (n - pi[n-1])
        # otherwise it's n.
        length_per = n - pi[n-1]
        if n % length_per == 0:
            return length_per
        else:
            return n

    idx=1  # index in input_data
    out=[]
    for _ in range(t):
        S = input_data[idx]; idx+=1
        X = input_data[idx]; idx+=1
        Y = input_data[idx]; idx+=1
        
        # Count zeros and ones in X, Y
        a = X.count('0')
        b = len(X) - a
        c = Y.count('0')
        d = len(Y) - c
        
        # Compute minimal period of S
        Ls = len(S)
        if Ls==1:
            # Then the minimal period length is 1
            # p = Ls // 1 = 1
            p = 1
        else:
            per = minimal_period_length(S)
            p = Ls // per  # S = g^p for some base g of length per

        # We want to find a nonnegative integer q satisfying:
        #   p*a + q*b = p*c + q*d   =>  q*(b-d) = p*(c-a).
        # Denote val = p*(c-a), den = (b-d).
        val = p*(c - a)
        den = b - d
        
        if den==0:
            # Then we need val==0 => p*(c-a)=0 => c-a=0 => c=a
            # If c!=a => no; else yes.
            if c==a:
                out.append("Yes")
            else:
                out.append("No")
        else:
            # We need val % den == 0 => q= val//den, and q>=0
            if val%den==0:
                q = val//den
                if q>=0:
                    out.append("Yes")
                else:
                    out.append("No")
            else:
                out.append("No")

    print("
".join(out))