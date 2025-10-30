def main():
    S = list(map(int, input().split()))
    
    # Check 1: non-decreasing
    if S != sorted(S):
        print("No")
        return
    
    # Check 2 and 3: each Si is between 100 and 675 (inclusive) and multiple of 25
    for x in S:
        if x < 100 or x > 675 or x % 25 != 0:
            print("No")
            return
    
    # If all checks pass
    print("Yes")

# Do not remove - this is required
if __name__ == "__main__":
    main()