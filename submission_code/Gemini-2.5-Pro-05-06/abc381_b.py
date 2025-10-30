import collections

def main():
    S = input()
    n = len(S)

    # Condition 1: The length of S must be even.
    # Text: "\lvert T vert is even."
    if n % 2 != 0:
        print("No")
        return

    # Condition 2: For each integer i satisfying 1 <= i <= |T|/2, 
    # the (2i-1)-th and 2i-th characters of T are equal.
    # In 0-based indexing, this means S[0]==S[1], S[2]==S[3], etc.
    # We loop `idx` from 0 to n-2 with a step of 2.
    # S[idx] corresponds to the (2k-1)-th character and S[idx+1] to the (2k)-th character
    # for some k (specifically, k = idx/2 + 1).
    for idx in range(0, n, 2): 
        if S[idx] != S[idx+1]:
            print("No")
            return
            
    # Condition 3: Each character appears in T exactly zero or two times. 
    # That is, every character contained in T appears exactly twice in T.
    # We count character frequencies. For every character present in S,
    # its count must be exactly 2.
    char_frequencies = collections.Counter(S)
    
    # Iterate through the counts of characters that are actually present in S.
    # Each of these counts must be 2.
    for count in char_frequencies.values():
        if count != 2:
            print("No")
            return
            
    # If all three conditions are satisfied
    print("Yes")

# Call the main function to execute the program.
main()