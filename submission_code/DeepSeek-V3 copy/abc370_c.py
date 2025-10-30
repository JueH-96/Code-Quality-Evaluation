# YOUR CODE HERE
def main():
    S = input().strip()
    T = input().strip()
    if S == T:
        print(0)
        return
    X = []
    current_S = list(S)
    target_T = list(T)
    n = len(S)
    for i in range(n):
        if current_S[i] != target_T[i]:
            current_S[i] = target_T[i]
            X.append(''.join(current_S))
    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    main()