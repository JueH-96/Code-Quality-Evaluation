# YOUR CODE HERE
def main():
    N = int(input())
    S = input().strip()
    total_T = S.count('T')
    total_A = S.count('A')
    if total_T > total_A:
        print('T')
    elif total_A > total_T:
        print('A')
    else:
        K = total_T  # same as total_A
        cum_T, cum_A = 0, 0
        for c in S:
            if c == 'T':
                cum_T += 1
            else:
                cum_A += 1
            if cum_T == K and cum_A < K:
                print('T')
                return
            if cum_A == K and cum_T < K:
                print('A')
                return
        # If somehow not returned yet, default to T
        print('T')

if __name__ == "__main__":
    main()