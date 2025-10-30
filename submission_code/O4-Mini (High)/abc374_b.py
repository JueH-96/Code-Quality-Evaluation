def main():
    S = input().strip()
    T = input().strip()
    
    # Compare character by character up to the longer length
    max_len = max(len(S), len(T))
    for i in range(max_len):
        # If one string ended or characters differ, report the 1-based position
        if i >= len(S) or i >= len(T) or S[i] != T[i]:
            print(i + 1)
            return
    
    # If no differences found, print 0
    print(0)

if __name__ == "__main__":
    main()