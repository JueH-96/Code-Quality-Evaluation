def main():
    S = list(map(int, input().split()))
    
    # Condition 1: Non-decreasing
    cond1 = all(S[i] <= S[i+1] for i in range(len(S)-1))
    # Condition 2: Each between 100 and 675 inclusive
    cond2 = all(100 <= x <= 675 for x in S)
    # Condition 3: Each is a multiple of 25
    cond3 = all(x % 25 == 0 for x in S)
    
    if cond1 and cond2 and cond3:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()