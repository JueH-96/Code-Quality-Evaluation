def is_11_22_string(S):
    N = len(S)
    
    # Check if the length is odd
    if N % 2 == 0:
        return False
    
    # Calculate the position of '/' (0-indexed)
    slash_pos = (N + 1) // 2 - 1
    
    # Check if the character at the calculated position is '/'
    if S[slash_pos] != '/':
        return False
    
    # Check if all characters before the '/' are '1's and all characters after the '/' are '2's
    return all(c == '1' for c in S[:slash_pos]) and all(c == '2' for c in S[slash_pos+1:])

def main():
    N = int(input())
    S = input()
    
    if is_11_22_string(S):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()