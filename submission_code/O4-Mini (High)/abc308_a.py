def main():
    S = list(map(int, input().split()))
    
    # Condition 1: non‐decreasing
    if any(S[i] > S[i+1] for i in range(7)):
        print("No")
        return
    
    # Condition 2 & 3: each between 100 and 675 inclusive and multiple of 25
    for x in S:
        if not (100 <= x <= 675 and x % 25 == 0):
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()