def main():
    S = input().rstrip()
    T = input().rstrip()
    
    # Find the smaller length
    min_len = min(len(S), len(T))
    
    # Compare each character up to the min length
    for i in range(min_len):
        if S[i] != T[i]:
            print(i + 1)
            return
    
    # If all matched up to min_len, difference is in length or they are equal
    if len(S) == len(T):
        print(0)  # They are exactly the same
    else:
        print(min_len + 1)  # The first differing position due to length

# Do not forget to call main()
main()