import itertools

def main():
    N, M = map(int, input().split())
    S_list = []
    for _ in range(N):
        S_list.append(input())

    # Function to check if two strings s1 and s2 (of length M) 
    # differ by exactly one character.
    # M is captured from the outer scope (main function's M).
    def check_diff_one(s1, s2):
        diff_count = 0
        for k in range(M): # M is the length of the strings
            if s1[k] != s2[k]:
                diff_count += 1
        
        return diff_count == 1

    # Iterate over all permutations of the S_list.
    # itertools.permutations(S_list) returns an iterator over tuples.
    # Each tuple 'p' is a permutation of the strings from S_list.
    for p in itertools.permutations(S_list):
        # p is a candidate sequence (T_1, T_2, ..., T_N)
        # In Python's 0-indexing, p[0] is T_1, p[1] is T_2, ..., p[N-1] is T_N.
        
        current_permutation_is_valid = True
        # We need to check N-1 pairs: (T_1,T_2), (T_2,T_3), ..., (T_{N-1},T_N).
        # These correspond to (p[0],p[1]), (p[1],p[2]), ..., (p[N-2],p[N-1])
        # The loop for i from 0 to N-2 covers these N-1 pairs.
        for i in range(N - 1): 
            if not check_diff_one(p[i], p[i+1]):
                current_permutation_is_valid = False
                break  # This permutation is not valid, try the next one
        
        if current_permutation_is_valid:
            # We found a permutation that satisfies the condition
            print("Yes")
            return # Exit the main function, and thus the program

    # If the loop finishes, it means no permutation satisfied the condition
    print("No")

if __name__ == '__main__':
    main()