# YOUR CODE HERE
def main():
    S_T = input().strip().split()
    if len(S_T) == 2:
        S, T = S_T
    else:
        S = S_T[0]
        T = ''
    n = len(S)
    m = len(T)
    for w in range(1, n):
        # Maximum possible c is min(w, m)
        for c in range(1, w+1):
            res = []
            for i in range(0, n, w):
                substring = S[i:i+w]
                if len(substring) >= c:
                    res.append(substring[c-1])
            candidate = ''.join(res)
            if candidate == T:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()