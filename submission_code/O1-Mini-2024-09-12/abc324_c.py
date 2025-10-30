# YOUR CODE HERE
import sys

def possible(T, S):
    len_T = len(T)
    len_S = len(S)
    if T == S:
        return True
    if len_T == len_S + 1:
        # Check if S can be obtained by deleting one character from T
        i = j = 0
        mismatch = False
        while i < len_T and j < len_S:
            if T[i] != S[j]:
                if mismatch:
                    return False
                mismatch = True
                i += 1
            else:
                i += 1
                j += 1
        return True
    if len_T +1 == len_S:
        # Check if T can be obtained by deleting one character from S
        return possible(S, T)
    if len_T == len_S:
        # Check if they differ by exactly one character
        diff = 0
        for a, b in zip(T, S):
            if a != b:
                diff +=1
                if diff >1:
                    return False
        return diff ==1
    return False

def main():
    import sys
    import sys
    input = sys.stdin.read().splitlines()
    first = input[0].split()
    N = int(first[0])
    T_prime = first[1]
    S_list = input[1:N+1]
    result = []
    for idx, S in enumerate(S_list, 1):
        if possible(T_prime, S):
            result.append(idx)
    print(len(result))
    if result:
        print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()