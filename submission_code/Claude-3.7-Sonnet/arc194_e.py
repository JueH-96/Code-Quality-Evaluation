def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def main():
    # Read input
    N, X, Y = map(int, input().strip().split())
    S = input().strip()
    T = input().strip()
    
    # Check if total number of 1's is the same
    if S.count('1') != T.count('1'):
        print("No")
        return
    
    # Calculate GCD of X and Y
    g = gcd(X, Y)
    
    # Check each equivalence class
    for c in range(g):
        count_S_1 = 0
        count_T_1 = 0
        
        for i in range(N):
            if i % (X+Y) % g == c:
                if S[i] == '1':
                    count_S_1 += 1
                if T[i] == '1':
                    count_T_1 += 1
        
        if count_S_1 != count_T_1:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()