def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1] if len(input_data) > 1 else ""
    
    min_len = min(len(S), len(T))
    
    # Check character by character in the overlapping portion of the strings
    for i in range(min_len):
        if S[i] != T[i]:
            print(i + 1)  # positions are 1-indexed
            return
    
    # If all characters in the overlap are equal, check if lengths are equal
    if len(S) == len(T):
        print(0)
    else:
        print(min_len + 1)

if __name__ == '__main__':
    main()